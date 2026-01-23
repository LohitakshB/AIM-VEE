"""Chemprop baseline training entrypoint."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

from aimvee.data_utils.data_prep import add_split_args, iter_split_csvs


def _resolve_chemprop_cmd() -> list[str]:
    cmd = shutil.which("chemprop")
    if cmd:
        return [cmd, "train"]
    cmd = shutil.which("chemprop_train")
    if cmd:
        return [cmd]
    return []


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Chemprop model on splits.", add_help=add_help
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss_function", type=str, default="mse")
    parser.add_argument("--metric", type=str, default="mae")
    parser.add_argument(
        "--chemprop-args",
        default=None,
        help="Extra args passed through to chemprop_train.",
    )
    return parser


def run_chemprop(args: argparse.Namespace) -> None:

    chemprop_cmd = _resolve_chemprop_cmd()
    if not chemprop_cmd:
        raise SystemExit(
            "Chemprop CLI not found. Install chemprop and ensure `chemprop` is on PATH."
        )

    split_iter = iter_split_csvs(
        xyz_dir=Path(args.xyz_dir),
        metadata_path=Path(args.metadata),
        output_root=Path(args.splits_output),
        id_column=args.id_column,
        target_column=args.target_column,
        smiles_column=args.smiles_column if args.smiles_column else None,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        split_method=args.split_method,
        train_all_splits=args.train_all_splits,
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

    extra_args = shlex.split(args.chemprop_args)

    for split_method, train_csv, val_csv, _ in split_iter:
        print(f"Using splits ({split_method})...")
        split_output = Path(args.output_dir) / split_method
        split_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            *chemprop_cmd,
            "--data_path",
            str(train_csv),
            "--separate_val_path",
            str(val_csv),
            "--dataset_type",
            "regression",
            "--save_dir",
            str(split_output),
            "--smiles_columns",
            "smiles",
            "--target_columns",
            "target",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--hidden_size",
            str(args.hidden_size),
            "--depth",
            str(args.depth),
            "--dropout",
            str(args.dropout),
            "--seed",
            str(args.seed),
        ]
        cmd.extend(extra_args)

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = build_parser()
    add_split_args(parser, default_split_method=None)
    parser.set_defaults(id_column="geometry", target_column="output")
    args = parser.parse_args()
    run_chemprop(args)


if __name__ == "__main__":
    main()
