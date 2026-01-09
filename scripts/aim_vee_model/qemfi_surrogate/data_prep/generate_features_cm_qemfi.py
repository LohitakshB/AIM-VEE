"""Generate Coulomb matrix features for QeMFi data."""

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

from aimvee.cli import run_generate_cm


def build_parser() -> argparse.ArgumentParser:
    """Build the feature generation argument parser."""
    default_input = REPO_ROOT / "data" / "QeMFI_all" / "QeMFi"
    default_output = REPO_ROOT / "data" / "QeMFI_all" / "QeMFi_cm"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(default_input))
    parser.add_argument("--output-dir", default=str(default_output))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_generate_cm(args)


if __name__ == "__main__":
    main()
