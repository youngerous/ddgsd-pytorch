import glob
import os
import random
import time
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import get_trn_val_loader, get_tst_loader
from utils import AverageMeter, DistillationLoss, accuracy


class Trainer:
    def __init__(self, hparams, model, scaler=None, resultwriter=None):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        if self.device == "cuda":
            self.model = nn.DataParallel(self.model)
        self.scaler = scaler

        # optimizer, scheduler
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # metric
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.kd_loss = DistillationLoss(temp=hparams.temperature)
        self.mmd_loss = nn.MSELoss(reduction="none")
        self.lmbda = hparams.lmbda
        self.mu = hparams.mu

        # dataloader
        self.trn_loader, self.val_loader = get_trn_val_loader(
            data_dir=hparams.dpath.strip(),
            batch_size=hparams.batch_size,
            valid_size=0.1,
            num_workers=hparams.workers,
            pin_memory=True,
            ddgsd=hparams.ddgsd,
        )
        self.tst_loader = get_tst_loader(
            data_dir=hparams.dpath.strip(),
            batch_size=hparams.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        # model saving hparam
        self.version = 0
        while True:
            self.save_path = os.path.join(hparams.ckpt_path, f"version-{self.version}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)
        self.global_step = 0
        self.global_val_loss = 1e5
        self.eval_step = hparams.eval_step
        with open(
            os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(hparams, outfile, default_flow_style=False, allow_unicode=True)

        # experiment result logging
        self.resultwriter = resultwriter
        self.best_result = {
            "version": self.version,
            "test_loss": 0,
            "top_1_error": 0,
            "top_5_error": 0,
        }

    def configure_optimizers(self) -> Tuple[optim.Optimizer, Optional[optim.Optimizer]]:
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_decay_step_size,
            gamma=self.hparams.lr_decay_gamma,
        )
        return optimizer, scheduler

    def save_checkpoint(self, epoch: int, val_loss: float, model: nn.Module) -> None:
        tqdm.write(
            f"Val loss decreased ({self.global_val_loss:.4f} → {val_loss:.4f}). Saving model ..."
        )
        new_path = os.path.join(
            self.save_path, f"best_model_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        )
        for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        self.global_val_loss = val_loss

    def fit(self) -> dict:
        for epoch in tqdm(range(self.hparams.epoch), desc="epoch"):
            tqdm.write(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.5f}")
            result = self._train_epoch(epoch)
            if result["val_loss"] < self.global_val_loss:
                self.save_checkpoint(epoch, result["val_loss"], self.model)
                self.best_result.update(
                    {
                        "best_val_loss": self.global_val_loss,
                        "best_val_epoch": epoch,
                    }
                )
            self.lr_scheduler.step()

        self.summarywriter.close()
        return self.best_result

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.trn_loader), desc="steps", total=len(self.trn_loader)
        ):
            if self.hparams.ddgsd:
                batch_flip, batch_crop = batch
                img_a, label = map(lambda x: x.to(self.device), batch_flip)
                img_b, same_label = map(lambda x: x.to(self.device), batch_crop)
                assert torch.equal(label, same_label), "label not mathing"

                if self.hparams.amp:
                    with torch.cuda.amp.autocast():
                        logit_a, logit_b = self.model(img_a), self.model(img_b)
                        ce_loss = self.ce_loss(logit_a, label) + self.ce_loss(
                            logit_b, label
                        )
                        kd_loss = self.kd_loss(logit_a, logit_b) + self.kd_loss(
                            logit_b, logit_a
                        )
                        mmd_loss = self.mmd_loss(logit_a, logit_b)
                else:
                    logit_a, logit_b = self.model(img_a), self.model(img_b)
                    ce_loss = self.ce_loss(logit_a, label) + self.ce_loss(
                        logit_b, label
                    )
                    kd_loss = self.kd_loss(logit_a, logit_b) + self.kd_loss(
                        logit_b, logit_a
                    )
                    mmd_loss = self.mmd_loss(logit_a, logit_b)

                self.optimizer.zero_grad()
                if self.hparams.amp:
                    with torch.cuda.amp.autocast():
                        loss = (
                            ce_loss
                            + self.lmbda * torch.mean(kd_loss, dim=1)
                            + self.mu * torch.mean(mmd_loss, dim=1)
                        ).mean()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = (
                        ce_loss
                        + self.lmbda * torch.mean(kd_loss, dim=1)
                        + self.mu * torch.mean(mmd_loss, dim=1)
                    ).mean()
                    loss.backward()
                    self.optimizer.step()

                train_loss.update(loss.item())

                self.global_step += 1
                if self.global_step % self.eval_step == 0:
                    tqdm.write(
                        f"[DDGSD Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                    )
            else:  # baseline
                img, label = map(lambda x: x.to(self.device), batch)

                if self.hparams.amp:
                    with torch.cuda.amp.autocast():
                        logit = self.model(img)
                        loss = self.ce_loss(logit, label).mean()
                else:
                    logit = self.model(img)
                    loss = self.ce_loss(logit, label).mean()

                self.optimizer.zero_grad()
                if self.hparams.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                train_loss.update(loss.item())

                self.global_step += 1
                if self.global_step % self.eval_step == 0:
                    tqdm.write(
                        f"[Baseline Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                    )

        train_loss = train_loss.avg
        val_loss = self.validate(epoch)

        # tensorboard writing
        self.summarywriter.add_scalars(
            "lr", {"lr": self.optimizer.param_groups[0]["lr"]}, epoch
        )
        self.summarywriter.add_scalars(
            "loss/epoch", {"val": val_loss, "train": train_loss}, epoch
        )
        tqdm.write(f"** global step: {self.global_step}, val loss: {val_loss:.3f}")

        return {"val_loss": val_loss}

    def validate(self, epoch: int) -> float:
        val_loss = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader), desc="val_steps", total=len(self.val_loader)
            ):
                img, label = map(lambda x: x.to(self.device), batch)
                if self.hparams.amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(img)
                        loss = self.ce_loss(pred, label).mean()
                else:
                    pred = self.model(img)
                    loss = self.ce_loss(pred, label).mean()
                val_loss.update(loss.item())

        return val_loss.avg

    def test(self, state_dict) -> dict:
        test_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.load_state_dict(state_dict)
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.tst_loader), desc="tst_steps", total=len(self.tst_loader)
            ):
                img, label = map(lambda x: x.to(self.device), batch)
                pred = self.model(img)

                loss = self.ce_loss(pred, label).mean()
                test_loss.update(loss.item())

                prec1, prec5 = accuracy(pred, label, topk=(1, 5))
                top1.update(prec1.item())
                top5.update(prec5.item())

        print()
        print(f"** Test Loss: {test_loss.avg:.4f}")
        print(f"** Top-1 Error Rate: {100 - top1.avg:.4f}%")
        print(f"** Top-5 Error Rate: {100 - top5.avg:.4f}%")
        print()
        return {
            "test_loss": test_loss.avg,
            "top_1_error": 100 - top1.avg,
            "top_5_error": 100 - top5.avg,
        }
