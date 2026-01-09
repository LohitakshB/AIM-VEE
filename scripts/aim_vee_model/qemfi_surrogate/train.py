"""Train the QeMFi surrogate using the CLI helper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aimvee.cli import run_train_qemfi


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser."""
    default_data = REPO_ROOT / "data" / "QeMFI_all" / "model_training_test_data"
    default_out = REPO_ROOT / "models" / "qemfi_surrogate"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(default_data))
    parser.add_argument("--output-dir", default=str(default_out))
    parser.add_argument("--method", default="CM")
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_train_qemfi(args)


if __name__ == "__main__":
    main()
