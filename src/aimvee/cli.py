"""Command-line entrypoints for AIM-VEE workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from aimvee.models import QemfiSurrogate
from aimvee.qemfi_utils.generate_cm import generate_cm
from aimvee.qemfi_utils.load_dataset import QemfiDataset
from aimvee.qemfi_utils.qemfi_prep import prep_data
from aimvee.qemfi_utils.train_utils import eval_epoch, train_epoch
from aimvee.utils import ensure_dir, select_device


def run_generate_cm(args: argparse.Namespace) -> None:
    """Generate Coulomb matrices for all QeMFi npz files in a folder."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files found in {input_dir}")

    for npz_path in npz_files:
        generate_cm(str(npz_path), str(output_dir))


def run_prep_qemfi(args: argparse.Namespace) -> None:
    """Prepare QeMFi datasets for surrogate training."""
    molecules = [m.strip() for m in args.molecules.split(",") if m.strip()]
    if not molecules:
        raise SystemExit("At least one molecule is required.")

    prep_data(
        data_dir=str(Path(args.data_dir)),
        molecules=molecules,
        reps_dir=str(Path(args.reps_dir)),
        qemfi_dir=str(Path(args.qemfi_dir)),
        method=args.method,
        n_geom_per_mol=args.n_geom_per_mol,
    )


def run_train_qemfi(args: argparse.Namespace) -> None:
    """Train the QeMFi surrogate model and save best checkpoint."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    x_train = np.load(data_dir / f"{args.method}_train.npy")
    y_train = np.load(data_dir / "EV_train.npy")
    x_val = np.load(data_dir / f"{args.method}_val.npy")
    y_val = np.load(data_dir / "EV_val.npy")

    mask_train = y_train >= 0
    mask_val = y_val >= 0
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    x_val = x_val[mask_val]
    y_val = y_val[mask_val]

    d_rep = x_train.shape[1] - 2
    x_train_rep = x_train[:, :d_rep]
    x_train_idx = x_train[:, d_rep:]
    x_val_rep = x_val[:, :d_rep]
    x_val_idx = x_val[:, d_rep:]

    scaler_x = StandardScaler()
    x_train_rep = scaler_x.fit_transform(x_train_rep)
    x_val_rep = scaler_x.transform(x_val_rep)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    target_scale = float(scaler_y.scale_[0])

    if args.pca_components:
        pca = PCA(n_components=args.pca_components)
        x_train_rep = pca.fit_transform(x_train_rep)
        x_val_rep = pca.transform(x_val_rep)
    else:
        pca = None

    x_train = np.concatenate([x_train_rep, x_train_idx], axis=1)
    x_val = np.concatenate([x_val_rep, x_val_idx], axis=1)

    train_ds = QemfiDataset(x_train, y_train)
    val_ds = QemfiDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = select_device(args.device)
    print(f"Using device: {device}")

    model = QemfiSurrogate(
        d_rep=train_ds.d_rep,
        n_fids=train_ds.n_fids,
        n_states=train_ds.n_states,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    joblib.dump(scaler_x, output_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, output_dir / "scaler_y.pkl")
    if pca is not None:
        joblib.dump(pca, output_dir / "pca_X.pkl")

    best_val = float("inf")
    cm_to_ev = 1.239841984e-4
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_mae = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            target_scale=target_scale,
        )
        val_mae = eval_epoch(
            model,
            val_loader,
            device=device,
            target_scale=target_scale,
        )

        scheduler.step(val_mae)

        train_mae_ev = train_mae * cm_to_ev
        val_mae_ev = val_mae * cm_to_ev
        print(
            "Epoch "
            f"{epoch:03d} "
            f"train_mae_cm={train_mae:.3f} "
            f"train_mae_ev={train_mae_ev:.6f} "
            f"val_mae_cm={val_mae:.3f} "
            f"val_mae_ev={val_mae_ev:.6f} "
            f"best_val_cm={best_val if best_val < float('inf') else val_mae:.3f}"
        )

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            best_val_ev = best_val * cm_to_ev
            print(
                f"New best: val_mae_cm={best_val:.3f} val_mae_ev={best_val_ev:.6f}"
            )

    
def build_parser() -> argparse.ArgumentParser:
    """Build the main CLI parser."""
    parser = argparse.ArgumentParser(prog="aimvee")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cm = subparsers.add_parser("generate-cm", help="Generate Coulomb matrices from QeMFi npz files.")
    cm.add_argument("--input-dir", required=True)
    cm.add_argument("--output-dir", required=True)
    cm.set_defaults(func=run_generate_cm)

    prep = subparsers.add_parser("prep-qemfi", help="Prepare QeMFi datasets for training.")
    prep.add_argument("--qemfi-dir", required=True)
    prep.add_argument("--reps-dir", required=True)
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--method", default="CM")
    prep.add_argument(
        "--molecules",
        default="urea,acrolein,alanine,sma,nitrophenol,urocanic,dmabn,thymine,o-hbdi",
    )
    prep.add_argument("--n-geom-per-mol", type=int, default=15000)
    prep.set_defaults(func=run_prep_qemfi)

    train = subparsers.add_parser("train-qemfi", help="Train the QeMFi surrogate model.")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--method", default="CM")
    train.add_argument("--pca-components", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--emb-dim", type=int, default=32)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--device", default="")
    train.set_defaults(func=run_train_qemfi)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
