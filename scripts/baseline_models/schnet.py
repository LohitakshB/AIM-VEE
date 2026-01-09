"""Train SchNet baseline on QM9-style XYZ data using torch-geometric."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, radius_graph

repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "pyproject.toml").exists():
    repo_root = repo_root.parent
src_root = repo_root / "src"
if src_root.exists() and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from aimvee.qm9_utils.train_utils import (  # noqa: E402
    Qm9CsvDataset,
    build_schnet_parser,
    eval_epoch,
    select_device,
    train_epoch,
)
from aimvee.utils import ensure_dir  # noqa: E402


def main() -> None:
    parser = build_schnet_parser()
    args = parser.parse_args()

    print("Loading datasets...")
    train_ds = Qm9CsvDataset(Path(args.train_csv))
    val_ds = Qm9CsvDataset(Path(args.val_csv))

    print("Building data loaders...")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = select_device(args.device)
    print(f"Using device: {device}")

    def interaction_graph(
        pos: torch.Tensor, batch: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_cpu = pos.detach().cpu()
        batch_cpu = batch.detach().cpu() if batch is not None else None
        edge_index = radius_graph(
            pos_cpu,
            r=args.cutoff,
            batch=batch_cpu,
            loop=False,
            max_num_neighbors=args.max_num_neighbors,
        )
        row, col = edge_index
        edge_weight = (pos_cpu[row] - pos_cpu[col]).pow(2).sum(dim=-1).sqrt()
        return edge_index.to(device), edge_weight.to(device)

    model = SchNet(
        hidden_channels=args.hidden_dim,
        num_filters=args.num_filters,
        num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
        readout=args.readout,
        dipole=args.dipole,
        interaction_graph=interaction_graph,
    ).to(device)
    print("Model initialized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    best_val = math.inf
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_mae = train_epoch(model, train_loader, optimizer, device)
        val_mae = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} train_mae={train_mae:.6f} val_mae={val_mae:.6f}")
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"New best: val_mae={best_val:.6f}")

    torch.save(model.state_dict(), output_dir / "final_model.pt")


if __name__ == "__main__":
    main()
