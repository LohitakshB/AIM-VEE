"""Train the UMFF-MLP (Uncertainty-aware MFF-MLP) with deep ensembles."""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, global_mean_pool

from aimvee.data_utils.data_prep import add_split_args, iter_split_csvs
from aimvee.datasets.geometry import GeometryQemfiDataset
from aimvee.features.coulomb import build_cm_reps
from aimvee.models import MFFMLP, QemfiSurrogate
from aimvee.utils import ensure_dir, select_device


def _load_geometry_rows(csv_path: Path) -> List[Tuple[Path, float]]:
    rows: List[Tuple[Path, float]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xyz_path = row.get("xyz_path", "").strip()
            target = row.get("target", "").strip()
            if not xyz_path or target == "":
                continue
            rows.append((Path(xyz_path), float(target)))
    if not rows:
        raise ValueError(f"No usable rows found in {csv_path}")
    return rows


CM_SIZE = 22


def _pick_schnet_latent_layer(model: SchNet) -> Tuple[str, torch.nn.Module, int]:
    for name in ("lin3", "lin2", "lin1"):
        layer = getattr(model, name, None)
        if layer is not None and hasattr(layer, "in_features"):
            return name, layer, int(layer.in_features)
    raise AttributeError("SchNet has no linear layers to attach latent hook.")


def _capture_schnet_latent(layer: torch.nn.Module):
    handle_latent: dict = {}

    def hook(_module, inputs, _output):
        handle_latent["value"] = inputs[0]

    hook_handle = layer.register_forward_hook(hook)
    return handle_latent, hook_handle


def _predict_qemfi_energies(
    model: QemfiSurrogate,
    cm_reps: np.ndarray,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    pca,
    fid_idx: int,
    n_states: int,
    n_lowest: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x_scaled = scaler_x.transform(cm_reps)
    if pca is not None:
        x_scaled = pca.transform(x_scaled)

    n_geom = x_scaled.shape[0]

    rep_expanded = np.repeat(x_scaled, n_states, axis=0)
    fid_ids = np.full((n_geom * n_states,), fid_idx, dtype=np.int64)
    state_ids = np.tile(np.arange(n_states, dtype=np.int64), n_geom)

    feats = torch.tensor(rep_expanded, dtype=torch.float32)
    fid = torch.tensor(fid_ids, dtype=torch.long)
    state = torch.tensor(state_ids, dtype=torch.long)

    preds_cpu: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, feats.shape[0], batch_size):
            end = start + batch_size
            batch_feats = feats[start:end].to(device)
            batch_fid = fid[start:end].to(device)
            batch_state = state[start:end].to(device)
            preds_cpu.append(model(batch_feats, batch_fid, batch_state).cpu())

    preds_np = torch.cat(preds_cpu, dim=0).numpy().reshape(-1, 1)
    preds_cm = scaler_y.inverse_transform(preds_np).reshape(n_geom, n_states)
    preds_ev = preds_cm * 1.239841984e-4

    sorted_preds = np.sort(preds_ev, axis=1)
    n_keep = min(n_lowest, sorted_preds.shape[1])
    return sorted_preds[:, :n_keep].astype(np.float32)


def _assemble_features(
    batch,
    schnet: SchNet,
    latent_handle: dict,
) -> torch.Tensor:
    latent_handle.clear()
    _ = schnet(batch.z, batch.pos, batch.batch)
    latent = latent_handle.get("value")
    if latent is None:
        raise RuntimeError("Failed to capture SchNet latents.")
    if latent.dim() == 1:
        latent = latent.unsqueeze(1)
    if latent.shape[0] != batch.num_graphs:
        latent = global_mean_pool(latent, batch.batch)
    qemfi = batch.qemfi
    if qemfi.dim() == 1:
        qemfi = qemfi.unsqueeze(1)
    return torch.cat([qemfi, latent], dim=1)


def _train_epoch(
    schnet: SchNet,
    mff_mlp: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    latent_handle: dict,
) -> float:
    schnet.train()
    mff_mlp.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        features = _assemble_features(batch, schnet, latent_handle)
        preds = mff_mlp(features)
        loss = torch.nn.functional.l1_loss(preds, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def _eval_epoch(
    schnet: SchNet,
    mff_mlp: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    latent_handle: dict,
) -> float:
    schnet.eval()
    mff_mlp.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features = _assemble_features(batch, schnet, latent_handle)
            preds = mff_mlp(features)
            loss = torch.nn.functional.l1_loss(preds, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def _ensemble_predictions(
    ensemble: List[Tuple[SchNet, torch.nn.Module]],
    latent_handles: List[dict],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for schnet, mff in ensemble:
        schnet.eval()
        mff.eval()
    preds_per_member: List[List[torch.Tensor]] = [
        [] for _ in range(len(ensemble))
    ]
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            targets.append(batch.y.view(-1).cpu())
            for idx, (schnet, mff) in enumerate(ensemble):
                features = _assemble_features(batch, schnet, latent_handles[idx])
                preds_per_member[idx].append(mff(features).cpu())
    member_preds = [torch.cat(chunks, dim=0) for chunks in preds_per_member]
    pred_stack = torch.stack(member_preds, dim=0)
    targets_np = torch.cat(targets, dim=0).numpy()
    pred_mean = pred_stack.mean(dim=0).numpy()
    pred_std = pred_stack.std(dim=0, unbiased=False).numpy()
    return targets_np, pred_mean, pred_std


def _save_prediction_csv(
    path: Path,
    targets: np.ndarray,
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
    calibrated_std: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["target", "pred_mean", "pred_std", "pred_std_calibrated"]
        )
        for target, pred, std, std_cal in zip(
            targets, mean_preds, std_preds, calibrated_std
        ):
            writer.writerow([target, pred, std, std_cal])


def _repo_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train UMFF-MLP with deep ensembles + isotonic calibration on "
            "QeMFi + SchNet features."
        ),
        add_help=add_help,
    )
    add_split_args(parser, default_split_method=None)
    parser.set_defaults(id_column="geometry", target_column="output")

    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)

    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--ensemble-seed", type=int, default=0)

    parser.add_argument("--qemfi-model-dir", required=True)
    parser.add_argument("--qemfi-fidelity", type=int, default=4)
    parser.add_argument("--qemfi-batch-size", type=int, default=2048)
    parser.add_argument("--n-lowest-states", type=int, default=10)

    parser.add_argument("--schnet-hidden-dim", type=int, default=128)
    parser.add_argument("--schnet-num-filters", type=int, default=128)
    parser.add_argument("--schnet-num-interactions", type=int, default=6)
    parser.add_argument("--schnet-num-gaussians", type=int, default=50)
    parser.add_argument("--schnet-cutoff", type=float, default=10.0)
    parser.add_argument("--schnet-max-num-neighbors", type=int, default=32)
    parser.add_argument("--schnet-readout", choices=("add", "mean"), default="add")
    parser.add_argument("--schnet-dipole", action="store_true")
    parser.add_argument("--schnet-batch-size", type=int, default=32)

    parser.add_argument("--device", default="")
    return parser


def _load_qemfi_assets(model_dir: Path):
    model_path = model_dir / "best_model.pt"
    scaler_x_path = model_dir / "scaler_X.pkl"
    scaler_y_path = model_dir / "scaler_y.pkl"
    pca_path = model_dir / "pca_X.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing QeMFi model: {model_path}")
    if not scaler_x_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError(
            f"Missing QeMFi scalers in {model_dir}. Expected scaler_X.pkl/scaler_y.pkl."
        )

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    pca = joblib.load(pca_path) if pca_path.exists() else None

    raw_d_rep = scaler_x.mean_.shape[0]
    return model_path, scaler_x, scaler_y, pca, raw_d_rep


def _infer_qemfi_dims(state_dict: dict) -> Tuple[int, int, int, int, int, float]:
    fid_weight = state_dict["fid_emb.weight"]
    state_weight = state_dict["state_emb.weight"]
    emb_dim = fid_weight.shape[1]
    n_fids = fid_weight.shape[0]
    n_states = state_weight.shape[0]

    input_weight = state_dict["input_proj.0.weight"]
    hidden_dim = input_weight.shape[0]
    in_dim = input_weight.shape[1]
    d_rep = in_dim - 2 * emb_dim
    if d_rep <= 0:
        raise ValueError(f"Invalid QeMFi input dimension inferred: {d_rep}")

    dropout = 0.05
    return d_rep, n_fids, n_states, hidden_dim, emb_dim, dropout


def _load_qemfi_model(
    model_path: Path, device: torch.device
) -> Tuple[QemfiSurrogate, int, int, int]:
    state_dict = torch.load(model_path, map_location="cpu")
    d_rep, n_fids, n_states, hidden_dim, emb_dim, dropout = _infer_qemfi_dims(
        state_dict
    )

    model = QemfiSurrogate(
        d_rep=d_rep,
        n_fids=n_fids,
        n_states=n_states,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, d_rep, n_fids, n_states


def _build_schnet_model(args: argparse.Namespace, device: torch.device) -> SchNet:
    model = SchNet(
        hidden_channels=args.schnet_hidden_dim,
        num_filters=args.schnet_num_filters,
        num_interactions=args.schnet_num_interactions,
        num_gaussians=args.schnet_num_gaussians,
        cutoff=args.schnet_cutoff,
        max_num_neighbors=args.schnet_max_num_neighbors,
        readout=args.schnet_readout,
        dipole=args.schnet_dipole,
    ).to(device)
    return model


def _seed_ensemble_member(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_umff_mlp(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    device = select_device(args.device)
    print(f"Using device: {device}")

    if args.ensemble_size < 2:
        raise ValueError("ensemble-size must be >= 2 for uncertainty estimates.")

    qemfi_model_dir = Path(args.qemfi_model_dir)
    model_path, scaler_x, scaler_y, pca, raw_d_rep = _load_qemfi_assets(
        qemfi_model_dir
    )
    cm_size = CM_SIZE

    qemfi_model, model_d_rep, n_fids, n_states = _load_qemfi_model(
        model_path, device
    )

    if pca is not None:
        if pca.components_.shape[0] != model_d_rep:
            raise ValueError(
                "QeMFi PCA output dimension does not match model input. "
                f"Expected {model_d_rep}, got {pca.components_.shape[0]}."
            )
    elif model_d_rep != raw_d_rep:
        raise ValueError(
            "QeMFi model input dimension does not match raw CM size. "
            f"Expected {model_d_rep}, got {raw_d_rep}."
        )

    if args.qemfi_fidelity < 0 or args.qemfi_fidelity >= n_fids:
        raise ValueError(
            f"Fidelity index {args.qemfi_fidelity} out of range (0..{n_fids - 1})."
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
        split_name=args.split_name,
        predefined_train_csv=(
            Path(args.predefined_train) if args.predefined_train else None
        ),
        predefined_val_csv=(Path(args.predefined_val) if args.predefined_val else None),
        predefined_test_csv=(
            Path(args.predefined_test) if args.predefined_test else None
        ),
    )

    for split_method, train_csv, val_csv, split_output in split_iter:
        print(f"Using splits ({split_method})...")
        test_csv = split_output / "test.csv"
        if not test_csv.exists():
            raise FileNotFoundError(f"Missing test split CSV: {test_csv}")

        train_rows = _load_geometry_rows(train_csv)
        val_rows = _load_geometry_rows(val_csv)
        test_rows = _load_geometry_rows(test_csv)

        print("Generating QeMFi surrogate energies...")
        train_cm = build_cm_reps(train_rows, cm_size)
        val_cm = build_cm_reps(val_rows, cm_size)
        test_cm = build_cm_reps(test_rows, cm_size)

        train_qemfi = _predict_qemfi_energies(
            qemfi_model,
            train_cm,
            scaler_x,
            scaler_y,
            pca,
            args.qemfi_fidelity,
            n_states,
            args.n_lowest_states,
            args.qemfi_batch_size,
            device,
        )
        val_qemfi = _predict_qemfi_energies(
            qemfi_model,
            val_cm,
            scaler_x,
            scaler_y,
            pca,
            args.qemfi_fidelity,
            n_states,
            args.n_lowest_states,
            args.qemfi_batch_size,
            device,
        )
        test_qemfi = _predict_qemfi_energies(
            qemfi_model,
            test_cm,
            scaler_x,
            scaler_y,
            pca,
            args.qemfi_fidelity,
            n_states,
            args.n_lowest_states,
            args.qemfi_batch_size,
            device,
        )

        qemfi_scaler = StandardScaler()
        train_qemfi = qemfi_scaler.fit_transform(train_qemfi)
        val_qemfi = qemfi_scaler.transform(val_qemfi)
        test_qemfi = qemfi_scaler.transform(test_qemfi)

        train_ds = GeometryQemfiDataset(train_rows, train_qemfi)
        val_ds = GeometryQemfiDataset(val_rows, val_qemfi)
        test_ds = GeometryQemfiDataset(test_rows, test_qemfi)

        train_loader = DataLoader(
            train_ds, batch_size=args.schnet_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.schnet_batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.schnet_batch_size, shuffle=False
        )

        run_output = output_root / split_method
        ensure_dir(run_output)
        joblib.dump(qemfi_scaler, run_output / "qemfi_scaler.pkl")

        ensemble: List[Tuple[SchNet, torch.nn.Module]] = []

        for member in range(args.ensemble_size):
            _seed_ensemble_member(args.ensemble_seed + member)
            schnet_model = _build_schnet_model(args, device)
            latent_name, latent_layer, latent_dim = _pick_schnet_latent_layer(
                schnet_model
            )
            mff_mlp = MFFMLP(
                in_dim=train_qemfi.shape[1] + latent_dim,
                hidden_dim=args.mlp_hidden_dim,
                num_layers=args.mlp_layers,
                dropout=args.mlp_dropout,
            ).to(device)

            optimizer = torch.optim.Adam(
                list(schnet_model.parameters()) + list(mff_mlp.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            member_output = run_output / f"ensemble_{member:02d}"
            ensure_dir(member_output)

            print(
                f"Ensemble member {member + 1}/{args.ensemble_size} "
                f"using SchNet latent from {latent_name} (dim={latent_dim})."
            )
            latent_handle, hook_handle = _capture_schnet_latent(latent_layer)
            best_val = float("inf")
            best_state = None
            print("Starting MFF-MLP training...")
            for epoch in range(1, args.epochs + 1):
                train_mae = _train_epoch(
                    schnet_model,
                    mff_mlp,
                    train_loader,
                    optimizer,
                    device,
                    latent_handle,
                )
                val_mae = _eval_epoch(
                    schnet_model,
                    mff_mlp,
                    val_loader,
                    device,
                    latent_handle,
                )
                print(
                    "Epoch "
                    f"{epoch:03d} "
                    f"train_mae={train_mae:.6f} "
                    f"val_mae={val_mae:.6f}"
                )
                if val_mae < best_val:
                    best_val = val_mae
                    best_state = (
                        copy.deepcopy(schnet_model.state_dict()),
                        copy.deepcopy(mff_mlp.state_dict()),
                    )
                    torch.save(mff_mlp.state_dict(), member_output / "best_model.pt")
                    torch.save(
                        schnet_model.state_dict(), member_output / "best_schnet.pt"
                    )
                    print(f"New best: val_mae={best_val:.6f}")

            hook_handle.remove()

            if best_state is not None:
                schnet_model.load_state_dict(best_state[0])
                mff_mlp.load_state_dict(best_state[1])

            ensemble.append((schnet_model, mff_mlp))

        latent_handles: List[dict] = []
        hook_handles = []
        for schnet_model, _ in ensemble:
            _, latent_layer, _ = _pick_schnet_latent_layer(schnet_model)
            latent_handle, hook_handle = _capture_schnet_latent(latent_layer)
            latent_handles.append(latent_handle)
            hook_handles.append(hook_handle)

        val_targets, val_mean, val_std = _ensemble_predictions(
            ensemble, latent_handles, val_loader, device
        )
        val_abs_err = np.abs(val_targets - val_mean)
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(val_std, val_abs_err)
        joblib.dump(iso_reg, run_output / "isotonic_reg.pkl")

        test_targets, test_mean, test_std = _ensemble_predictions(
            ensemble, latent_handles, test_loader, device
        )
        test_std_cal = iso_reg.predict(test_std)
        _save_prediction_csv(
            run_output / "test_predictions.csv",
            test_targets,
            test_mean,
            test_std,
            test_std_cal,
        )


        for handle in hook_handles:
            handle.remove()



def main() -> None:
    args = build_parser().parse_args()
    run_umff_mlp(args)


if __name__ == "__main__":
    main()
