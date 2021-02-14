import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--dpath", type=str, default="./data/")
    parser.add_argument("--ckpt-path", type=str, default="./checkpoints/")
    parser.add_argument("--result-path", type=str, default="./results.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--eval-step", type=int, default=100)
    parser.add_argument(
        "--amp", action="store_true", default=False, help="PyTorch(>=1.6.x) AMP"
    )

    parser.add_argument("--contain-test", action="store_true", default=False)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--ddgsd",
        action="store_true",
        help="Whether to apply ddgsd",
    )

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lr-decay-step-size", type=int, default=60)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.1)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Softmax temperature"
    )
    parser.add_argument(
        "--lmbda", type=float, default=1.0, help="Regularizing distillation loss"
    )
    parser.add_argument(
        "--mu", type=float, default=0.0005, help="Regularizing MMD loss"
    )

    args = parser.parse_args()
    return args