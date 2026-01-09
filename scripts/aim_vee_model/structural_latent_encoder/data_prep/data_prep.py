"""Prepare QM9-style data splits for the structural latent encoder pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aimvee.qm9_utils.qm9_prep import prepare_qm9_dataset


def _build_metadata_from_exc_ss(exc_dir: Path, output_csv: Path) -> None:
    """Create a metadata CSV by reading the first target per .dat file."""
    if not exc_dir.exists():
        raise FileNotFoundError(f"Excited-state directory not found: {exc_dir}")

    rows = []
    for dat_path in sorted(exc_dir.glob("*.dat")):
        geometry_id = dat_path.stem
        with dat_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    target = float(line)
                except ValueError:
                    continue
                rows.append((geometry_id, target))
                break

    if not rows:
        raise ValueError(f"No targets found in {exc_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "lowest_excited_state"])
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the data prep argument parser."""
    default_root = REPO_ROOT / "data" / "QM9GWBSE"
    default_xyz = default_root / "QM9_xyz_files"
    default_exc = default_root / "E_exc_SS"
    default_metadata = default_root / "metadata_e_exc_ss.csv"
    default_output = default_root / "qm9_splits"

    parser = argparse.ArgumentParser(
        prog="qm9_data_prep",
        description="Prepare QM9-like XYZ data with Bemis-Murcko scaffold splits.",
    )
    parser.add_argument("--xyz-dir", default=str(default_xyz))
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument(
        "--exc-ss-dir",
        default=str(default_exc),
        help="Directory of E_exc_SS .dat files used to auto-build metadata.csv.",
    )
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--target-column", default="lowest_excited_state")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        _build_metadata_from_exc_ss(Path(args.exc_ss_dir), metadata_path)

    prepare_qm9_dataset(
        xyz_dir=Path(args.xyz_dir),
        metadata_path=metadata_path,
        output_dir=Path(args.output_dir),
        id_column=args.id_column,
        target_column=args.target_column,
        smiles_column=args.smiles_column if args.smiles_column else None,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
