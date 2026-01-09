"""Prepare QeMFi datasets using the CLI helper."""

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

from aimvee.cli import run_prep_qemfi


def build_parser() -> argparse.ArgumentParser:
    """Build the data preparation argument parser."""
    default_qemfi = REPO_ROOT / "data" / "QeMFI_all" / "QeMFi"
    default_reps = REPO_ROOT / "data" / "QeMFI_all" / "QeMFi_cm"
    default_data = REPO_ROOT / "data" / "QeMFI_all" / "model_training_test_data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--qemfi-dir", default=str(default_qemfi))
    parser.add_argument("--reps-dir", default=str(default_reps))
    parser.add_argument("--data-dir", default=str(default_data))
    parser.add_argument("--method", default="CM")
    parser.add_argument(
        "--molecules",
        default="urea,acrolein,alanine,sma,nitrophenol,urocanic,dmabn,thymine,o-hbdi",
    )
    parser.add_argument("--n-geom-per-mol", type=int, default=15000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_prep_qemfi(args)


if __name__ == "__main__":
    main()
