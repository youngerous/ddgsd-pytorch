import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # default hparams
    parser.add_argument("--comment", default="실험내용기록")  #
    parser.add_argument("--dset", type=str, default="DATASET NAME")  #
    parser.add_argument("--dpath", type=str, default="DATASET PATH")  #
    parser.add_argument("--ckpt-path", type=str, default="./checkpoints/")
    parser.add_argument("--result-path", type=str, default="./results.csv")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--eval-step", type=int, default=100)
    parser.add_argument(
        "--amp", action="store_true", default=False, help="PyTorch(>=1.6.x) AMP"
    )
    parser.add_argument("--test", action="store_true", default=False)  #

    # training hparams
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", type=str, default="MODEL NAME")  #

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lr-decay-step-size", type=int, default=60)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.1)

    args = parser.parse_args()
    return args