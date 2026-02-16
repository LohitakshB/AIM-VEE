"""Dataset split preparation entrypoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aimvee.data_utils.data_prep import iter_split_csvs


def build_metadata_from_exc_ss(
    exc_dir: Path, xyz_dir: Path, output_csv: Path
) -> None:
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
                xyz_path = xyz_dir / f"{geometry_id}.xyz"
                rows.append((geometry_id, str(xyz_path), target))
                break

    if not rows:
        raise ValueError(f"No targets found in {exc_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "xyz_path", "lowest_excited_state"])
        writer.writerows(rows)


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    default_root = Path("data") / "QM9GWBSE"
    default_xyz = default_root / "QM9_xyz_files"
    default_exc = default_root / "E_exc_SS"
    default_metadata = default_root / "metadata_e_exc_ss.csv"
    default_output = default_root / "splits"

    parser = argparse.ArgumentParser(
        prog="data_prep",
        description="Prepare XYZ data with configurable split strategies.",
        add_help=add_help,
    )
    parser.add_argument("--xyz-dir", default=str(default_xyz))
    parser.add_argument("--metadata", default=str(default_metadata))
    parser.add_argument(
        "--splits-output",
        dest="splits_output",
        default=str(default_output),
        help="Output directory for splits.",
    )
    parser.add_argument(
        "--output-dir",
        dest="splits_output",
        help=argparse.SUPPRESS,
    )
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
    split_mode = parser.add_mutually_exclusive_group()
    split_mode.add_argument(
        "--train-all-splits",
        action="store_true",
        default=True,
        help="Train sequentially on all supported split methods (default).",
    )
    split_mode.add_argument(
        "--single-split",
        dest="train_all_splits",
        action="store_false",
        help="Train only on the selected --split-method.",
    )
    parser.add_argument(
        "--split-method",
        default=None,
        choices=(
            "bemis-murcko",
            "bemis_murcko",
            "random",
            "size",
            "tail-split",
            "predefined",
            "pre-defined",
        ),
        help="Split strategy for train/val/test generation.",
    )
    parser.add_argument(
        "--split-name",
        help="Override the output subdirectory name for the split (default: split method).",
    )
    parser.add_argument(
        "--predefined-train",
        help="Path to a pre-defined train CSV (required for --split-method predefined).",
    )
    parser.add_argument(
        "--predefined-val",
        help="Path to a pre-defined val CSV (required for --split-method predefined).",
    )
    parser.add_argument(
        "--predefined-test",
        help="Optional path to a pre-defined test CSV.",
    )
    return parser


def run_data_prep(args: argparse.Namespace) -> None:
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        build_metadata_from_exc_ss(
            Path(args.exc_ss_dir), Path(args.xyz_dir), metadata_path
        )

    list(
        iter_split_csvs(
            xyz_dir=Path(args.xyz_dir),
            metadata_path=metadata_path,
            output_root=Path(args.splits_output),
            id_column=args.id_column,
            target_column=args.target_column,
            smiles_column=args.smiles_column if args.smiles_column else None,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            split_method=args.split_method,
            train_all_splits=args.train_all_splits,
            split_name=args.split_name,
            predefined_train_csv=(
                Path(args.predefined_train) if args.predefined_train else None
            ),
            predefined_val_csv=(
                Path(args.predefined_val) if args.predefined_val else None
            ),
            predefined_test_csv=(
                Path(args.predefined_test) if args.predefined_test else None
            ),
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_data_prep(args)


if __name__ == "__main__":
    main()
