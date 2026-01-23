"""Generate a metadata CSV from E_exc_SS .dat files."""

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

import aimvee


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve()
    while repo_root != repo_root.parent and not (repo_root / "pyproject.toml").exists():
        repo_root = repo_root.parent
    default_root = repo_root / "data" / "QM9GWBSE"
    default_exc = default_root / "E_exc_SS"
    default_xyz = default_root / "QM9_xyz_files"
    default_output = default_root / "metadata.csv"

    parser = argparse.ArgumentParser(
        prog="generate_metadata",
        description="Generate metadata CSV from E_exc_SS .dat files.",
    )
    parser.add_argument("--exc-ss-dir", default=str(default_exc))
    parser.add_argument("--xyz-dir", default=str(default_xyz))
    parser.add_argument("--output", default=str(default_output))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    aimvee.build_metadata_from_exc_ss(
        Path(args.exc_ss_dir), Path(args.xyz_dir), Path(args.output)
    )


if __name__ == "__main__":
    main()
