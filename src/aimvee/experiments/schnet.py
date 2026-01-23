"""SchNet baseline training entrypoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from aimvee.data_utils.data_prep import add_split_args, iter_split_csvs
from aimvee.datasets.geometry import GeometryCsvDataset
from aimvee.trainers.torch_geom import eval_epoch, train_epoch
from aimvee.utils import ensure_dir, select_device


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SchNet baseline on splits.", add_help=add_help
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--max-num-neighbors", type=int, default=32)
    parser.add_argument("--readout", choices=("add", "mean"), default="add")
    parser.add_argument("--dipole", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="")
    return parser


def run_schnet(args: argparse.Namespace) -> None:
    xyz_dir = Path(args.xyz_dir)
    metadata = Path(args.metadata)
    split_root = Path(args.splits_output)

    try:
        split_iter = iter_split_csvs(
            xyz_dir=xyz_dir,
            metadata_path=metadata,
            output_root=split_root,
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
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for split_method, train_csv, val_csv, _ in split_iter:
        print(f"Using splits ({split_method})...")

        print("Loading datasets...")
        train_ds = GeometryCsvDataset(train_csv)
        val_ds = GeometryCsvDataset(val_csv)

        print("Building data loaders...")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        device = select_device(args.device)
        print(f"Using device: {device}")

        model = SchNet(
            hidden_channels=args.hidden_dim,
            num_filters=args.num_filters,
            num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians,
            cutoff=args.cutoff,
            max_num_neighbors=args.max_num_neighbors,
            readout=args.readout,
            dipole=args.dipole,
        ).to(device)
        print("Model initialized.")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        run_output_dir = Path(args.output_dir) / split_method
        ensure_dir(run_output_dir)

        best_val = math.inf
        print(f"Starting training ({split_method})...")
        for epoch in range(1, args.epochs + 1):
            train_mae = train_epoch(model, train_loader, optimizer, device)
            val_mae = eval_epoch(model, val_loader, device)
            print(
                f"Epoch {epoch:03d} train_mae={train_mae:.6f} "
                f"val_mae={val_mae:.6f}"
            )
            if val_mae < best_val:
                best_val = val_mae
                torch.save(model.state_dict(), run_output_dir / "best_model.pt")
                print(f"New best: val_mae={best_val:.6f}")

        torch.save(model.state_dict(), run_output_dir / "final_model.pt")


def main() -> None:
    parser = build_parser()
    add_split_args(parser, default_split_method=None)
    parser.set_defaults(id_column="geometry", target_column="output")
    args = parser.parse_args()
    run_schnet(args)


if __name__ == "__main__":
    main()
