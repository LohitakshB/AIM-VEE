"""Evaluation and plotting utilities for QM9-GWBSE experiments."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, global_mean_pool

from aimvee.datasets.geometry import GeometryCsvDataset, GeometryQemfiDataset
from aimvee.features.coulomb import build_cm_reps
from aimvee.features.morgan import load_morgan_dataset
from aimvee.models import MFFMLP, QemfiSurrogate
from aimvee.utils import ensure_dir, select_device


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()


def build_evaluate_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate QM9-GWBSE test performance.",
        add_help=add_help,
    )
    parser.add_argument("--xyz-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--splits-output", required=True)
    parser.add_argument("--split-method", default="random")
    parser.add_argument("--models-root", default=str(REPO_ROOT / "models"))
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "outputs" / "qm9_testing")
    )
    parser.add_argument("--models", default="")
    parser.add_argument("--device", default="")

    parser.add_argument("--schnet-hidden-dim", type=int, default=128)
    parser.add_argument("--schnet-num-filters", type=int, default=128)
    parser.add_argument("--schnet-num-interactions", type=int, default=6)
    parser.add_argument("--schnet-num-gaussians", type=int, default=50)
    parser.add_argument("--schnet-cutoff", type=float, default=10.0)
    parser.add_argument("--schnet-max-num-neighbors", type=int, default=32)
    parser.add_argument("--schnet-readout", choices=("add", "mean"), default="add")
    parser.add_argument("--schnet-dipole", action="store_true")
    parser.add_argument("--schnet-batch-size", type=int, default=32)

    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)

    parser.add_argument("--chemprop-extra-args", default="")

    parser.add_argument(
        "--qemfi-model-dir",
        default=str(REPO_ROOT / "models" / "aim_vee_model" / "qemfi_surrogate"),
    )
    parser.add_argument("--qemfi-fidelity", type=int, default=4)
    parser.add_argument("--qemfi-batch-size", type=int, default=2048)
    parser.add_argument("--n-lowest-states", type=int, default=10)
    parser.add_argument("--cm-size", type=int, default=22)
    parser.add_argument(
        "--qemfi-scaler-path",
        default=str(REPO_ROOT / "models" / "mff_mlp" / "random" / "qemfi_scaler.pkl"),
    )
    parser.add_argument("--mff-mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mff-mlp-layers", type=int, default=3)
    parser.add_argument("--mff-mlp-dropout", type=float, default=0.1)
    return parser


def build_plot_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot QM9-GWBSE evaluation outputs.",
        add_help=add_help,
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--split-method", default="random")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-points", type=int, default=20000)
    return parser


def _load_split_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _load_geometry_rows(csv_path: Path) -> List[Tuple[Path, float]]:
    dataset = GeometryCsvDataset(csv_path)
    return list(dataset.rows)


def _extract_ids_targets(rows: Iterable[Dict[str, str]]) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    targets: List[float] = []
    for row in rows:
        geom_id = row.get("geometry_id") or row.get("id") or ""
        target = row.get("target", "").strip()
        if target == "":
            continue
        ids.append(geom_id)
        targets.append(float(target))
    if not targets:
        raise ValueError("No targets found in split CSV.")
    return ids, np.asarray(targets, dtype=np.float32)


def _write_predictions(
    output_path: Path, ids: List[str], targets: np.ndarray, preds: np.ndarray
) -> None:
    if preds.shape[0] != targets.shape[0]:
        raise ValueError("Prediction/target size mismatch.")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["geometry_id", "target", "prediction", "error", "abs_error"])
        for geom_id, target, pred in zip(ids, targets, preds):
            error = float(pred - target)
            writer.writerow(
                [
                    geom_id,
                    f"{target:.10f}",
                    f"{pred:.10f}",
                    f"{error:.10f}",
                    f"{abs(error):.10f}",
                ]
            )


def _metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(targets, preds)
    try:
        rmse = mean_squared_error(targets, preds, squared=False)
    except TypeError:
        # Older sklearn versions don't support squared=.
        rmse = mean_squared_error(targets, preds) ** 0.5
    r2 = r2_score(targets, preds)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


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


def _predict_schnet(
    test_csv: Path,
    model_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    dataset = GeometryCsvDataset(test_csv)
    loader = DataLoader(dataset, batch_size=args.schnet_batch_size, shuffle=False)
    model = _build_schnet_model(args, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_preds = model(batch.z, batch.pos, batch.batch).view(-1)
            preds.append(batch_preds.cpu().numpy())
    return np.concatenate(preds, axis=0)


def _predict_rf_morgan(
    test_csv: Path, model_path: Path, args: argparse.Namespace
) -> np.ndarray:
    x_test, _ = load_morgan_dataset(test_csv, args.morgan_radius, args.morgan_bits)
    model = joblib.load(model_path)
    return model.predict(x_test)


def _resolve_chemprop_predict_cmd() -> List[str]:
    cmd = shutil.which("chemprop")
    if cmd:
        return [cmd, "predict"]
    cmd = shutil.which("chemprop_predict")
    if cmd:
        return [cmd]
    return []


def _predict_chemprop(
    test_csv: Path,
    model_dir: Path,
    preds_path: Path,
    extra_args: str,
) -> np.ndarray:
    cmd = _resolve_chemprop_predict_cmd()
    if not cmd:
        raise SystemExit(
            "Chemprop CLI not found. Install chemprop and ensure `chemprop` is on PATH."
        )
    extra = shlex.split(extra_args) if extra_args else []
    cmd = [
        *cmd,
        "--test_path",
        str(test_csv),
        "--preds_path",
        str(preds_path),
        "--checkpoint_dir",
        str(model_dir),
        "--smiles_columns",
        "smiles",
    ]
    cmd.extend(extra)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Chemprop prediction failed with "
            f"exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not preds_path.exists():
        raise FileNotFoundError(
            "Chemprop did not write predictions to "
            f"{preds_path}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    preds: List[float] = []
    with preds_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No predictions produced in {preds_path}")
    header = rows[0]
    header_lower = [col.strip().lower() for col in header]
    pred_idx = None
    for idx, col in enumerate(header_lower):
        if "pred" in col:
            pred_idx = idx
            break
    if pred_idx is None and header_lower:
        # chemprop_predict may overwrite the target column with predictions.
        pred_idx = len(header_lower) - 1
    data_rows = rows[1:] if header_lower else rows
    for row in data_rows:
        if not row:
            continue
        if pred_idx >= len(row):
            raise ValueError(f"Prediction column not found in {preds_path}")
        preds.append(float(row[pred_idx]))
    return np.asarray(preds, dtype=np.float32)


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


def _predict_qemfi_energies(
    model: QemfiSurrogate,
    cm_reps: np.ndarray,
    scaler_x,
    scaler_y,
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


def _predict_mff_mlp(
    test_csv: Path,
    model_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    qemfi_model_dir = Path(args.qemfi_model_dir)
    qemfi_model_path = qemfi_model_dir / "best_model.pt"
    scaler_x_path = qemfi_model_dir / "scaler_X.pkl"
    scaler_y_path = qemfi_model_dir / "scaler_y.pkl"
    pca_path = qemfi_model_dir / "pca_X.pkl"

    if not qemfi_model_path.exists():
        raise FileNotFoundError(f"Missing QeMFi model: {qemfi_model_path}")
    if not scaler_x_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError(
            "Missing QeMFi scalers in "
            f"{qemfi_model_dir}. Expected scaler_X.pkl/scaler_y.pkl."
        )
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    pca = joblib.load(pca_path) if pca_path.exists() else None

    qemfi_model, model_d_rep, n_fids, n_states = _load_qemfi_model(
        qemfi_model_path, device
    )
    raw_d_rep = scaler_x.mean_.shape[0]
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

    rows = _load_geometry_rows(test_csv)
    cm_reps = build_cm_reps(rows, args.cm_size)
    test_qemfi = _predict_qemfi_energies(
        qemfi_model,
        cm_reps,
        scaler_x,
        scaler_y,
        pca,
        args.qemfi_fidelity,
        n_states,
        args.n_lowest_states,
        args.qemfi_batch_size,
        device,
    )

    qemfi_scaler_path = Path(args.qemfi_scaler_path)
    if not qemfi_scaler_path.exists():
        raise FileNotFoundError(f"Missing QeMFi scaler: {qemfi_scaler_path}")
    qemfi_scaler = joblib.load(qemfi_scaler_path)
    test_qemfi = qemfi_scaler.transform(test_qemfi)

    test_ds = GeometryQemfiDataset(rows, test_qemfi)
    test_loader = DataLoader(test_ds, batch_size=args.schnet_batch_size, shuffle=False)

    schnet_model = _build_schnet_model(args, device)
    schnet_path = model_dir / "best_schnet.pt"
    mff_path = model_dir / "best_model.pt"
    if not schnet_path.exists() or not mff_path.exists():
        raise FileNotFoundError(
            f"Missing MFF-MLP weights in {model_dir} (best_schnet.pt, best_model.pt)."
        )
    schnet_model.load_state_dict(torch.load(schnet_path, map_location=device))

    _latent_name, latent_layer, latent_dim = _pick_schnet_latent_layer(schnet_model)
    mff_mlp = MFFMLP(
        in_dim=test_qemfi.shape[1] + latent_dim,
        hidden_dim=args.mff_mlp_hidden_dim,
        num_layers=args.mff_mlp_layers,
        dropout=args.mff_mlp_dropout,
    ).to(device)
    mff_mlp.load_state_dict(torch.load(mff_path, map_location=device))

    preds: List[np.ndarray] = []
    latent_handle, hook_handle = _capture_schnet_latent(latent_layer)
    schnet_model.eval()
    mff_mlp.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            latent_handle.clear()
            _ = schnet_model(batch.z, batch.pos, batch.batch)
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
            features = torch.cat([qemfi, latent], dim=1)
            batch_preds = mff_mlp(features)
            preds.append(batch_preds.cpu().numpy())
    hook_handle.remove()
    return np.concatenate(preds, axis=0)


def _parse_model_list(value: str | None) -> List[str]:
    if not value:
        return ["schnet", "rf_morgan", "chemprop", "mff_mlp"]
    return [item.strip() for item in value.split(",") if item.strip()]


def run_evaluate_qm9(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    print(f"Using device: {device}")

    split_dir = Path(args.splits_output) / args.split_method
    test_csv = split_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test split CSV: {test_csv}")

    models_root = Path(args.models_root)
    output_root = Path(args.output_dir) / args.split_method
    ensure_dir(output_root)

    split_rows = _load_split_rows(test_csv)
    ids, targets = _extract_ids_targets(split_rows)

    metrics_summary: Dict[str, Dict[str, float]] = {}
    models = _parse_model_list(args.models)
    for model_name in models:
        model_output = output_root / model_name
        ensure_dir(model_output)

        if model_name == "schnet":
            model_path = models_root / "schnet" / args.split_method / "best_model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing SchNet model: {model_path}")
            preds = _predict_schnet(test_csv, model_path, args, device)
        elif model_name == "rf_morgan":
            model_path = models_root / "rf_morgan" / args.split_method / "rf_morgan.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing RF Morgan model: {model_path}")
            preds = _predict_rf_morgan(test_csv, model_path, args)
        elif model_name == "chemprop":
            model_dir = models_root / "chemprop" / args.split_method
            if not model_dir.exists():
                raise FileNotFoundError(f"Missing Chemprop model dir: {model_dir}")
            preds_path = model_output / "chemprop_preds.csv"
            preds = _predict_chemprop(
                test_csv, model_dir, preds_path, args.chemprop_extra_args
            )
        elif model_name == "mff_mlp":
            model_dir = models_root / "mff_mlp" / args.split_method
            if not model_dir.exists():
                raise FileNotFoundError(f"Missing MFF-MLP model dir: {model_dir}")
            preds = _predict_mff_mlp(test_csv, model_dir, args, device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pred_path = model_output / "predictions.csv"
        _write_predictions(pred_path, ids, targets, preds)
        metrics = _metrics(targets, preds)
        metrics_summary[model_name] = metrics
        (model_output / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(f"{model_name} metrics: {metrics}")

    (output_root / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "mae", "rmse", "r2"])
        for model_name, metrics in metrics_summary.items():
            writer.writerow([model_name, metrics["mae"], metrics["rmse"], metrics["r2"]])
    print(f"Saved metrics to {summary_path}")


def _load_predictions(pred_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    targets: List[float] = []
    preds: List[float] = []
    with pred_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row.get("target", "")
            pred = row.get("prediction", "")
            if target == "" or pred == "":
                continue
            targets.append(float(target))
            preds.append(float(pred))
    if not targets:
        raise ValueError(f"No predictions found in {pred_path}")
    return np.asarray(targets), np.asarray(preds)


def _downsample(targets: np.ndarray, preds: np.ndarray, max_points: int):
    if targets.shape[0] <= max_points:
        return targets, preds
    idx = np.random.RandomState(13).choice(
        targets.shape[0], size=max_points, replace=False
    )
    return targets[idx], preds[idx]


def _parity_plot(
    targets: np.ndarray, preds: np.ndarray, title: str, output_path: Path
) -> None:
    plt.figure(figsize=(6, 6))
    min_val = float(min(targets.min(), preds.min()))
    max_val = float(max(targets.max(), preds.max()))
    plt.scatter(targets, preds, s=10, alpha=0.3, color="#2E7D32", edgecolors="none")
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="#333333", linewidth=1)
    plt.xlabel("True (eV)")
    plt.ylabel("Predicted (eV)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _residual_plot(
    targets: np.ndarray, preds: np.ndarray, title: str, output_path: Path
) -> None:
    residuals = preds - targets
    plt.figure(figsize=(6, 4))
    plt.scatter(targets, residuals, s=10, alpha=0.3, color="#1565C0", edgecolors="none")
    plt.axhline(0.0, linestyle="--", color="#333333", linewidth=1)
    plt.xlabel("True (eV)")
    plt.ylabel("Residual (pred - true)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _error_hist(
    targets: np.ndarray, preds: np.ndarray, title: str, output_path: Path
) -> None:
    abs_err = np.abs(preds - targets)
    plt.figure(figsize=(6, 4))
    plt.hist(abs_err, bins=40, color="#6A1B9A", alpha=0.8)
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _metrics_bar(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    models = list(metrics.keys())
    mae = [metrics[m]["mae"] for m in models]
    rmse = [metrics[m]["rmse"] for m in models]
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, mae, width, label="MAE", color="#00838F")
    plt.bar(x + width / 2, rmse, width, label="RMSE", color="#F9A825")
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Error (eV)")
    plt.title("Model Error Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _error_latency_plot(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    *,
    error_key: str = "rmse",
    latency_key: str = "latency_ms",
) -> None:
    points = []
    for model, values in metrics.items():
        if error_key not in values or latency_key not in values:
            continue
        points.append((model, values[latency_key], values[error_key]))
    if not points:
        return
    plt.figure(figsize=(6, 4))
    for model, latency_ms, err in points:
        plt.scatter(latency_ms, err, s=60, alpha=0.8, color="#5D4037", edgecolors="none")
        plt.text(latency_ms, err, f" {model}", va="center", fontsize=9)
    plt.xlabel("Latency per sample (ms)")
    plt.ylabel(f"{error_key.upper()} (eV)")
    plt.title("Error vs Latency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_plot_qm9(args: argparse.Namespace) -> None:
    split_dir = Path(args.results_dir) / args.split_method
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split results: {split_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else split_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}
    for model_dir in sorted(split_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        pred_path = model_dir / "predictions.csv"
        metrics_path = model_dir / "metrics.json"
        if not pred_path.exists():
            continue
        targets, preds = _load_predictions(pred_path)
        targets_ds, preds_ds = _downsample(targets, preds, args.max_points)

        model_name = model_dir.name
        if metrics_path.exists():
            metrics[model_name] = json.loads(metrics_path.read_text(encoding="utf-8"))

        _parity_plot(
            targets_ds,
            preds_ds,
            f"{model_name}: Parity Plot",
            output_dir / f"parity_{model_name}.png",
        )
        _residual_plot(
            targets_ds,
            preds_ds,
            f"{model_name}: Residuals",
            output_dir / f"residuals_{model_name}.png",
        )
        _error_hist(
            targets,
            preds,
            f"{model_name}: Absolute Error",
            output_dir / f"abs_error_{model_name}.png",
        )

    if metrics:
        _metrics_bar(metrics, output_dir / "metrics_summary.png")
        _error_latency_plot(metrics, output_dir / "error_latency.png")

    print(f"Saved plots to {output_dir}")


def main_evaluate() -> None:
    parser = build_evaluate_parser()
    args = parser.parse_args()
    run_evaluate_qm9(args)


def main_plot() -> None:
    parser = build_plot_parser()
    args = parser.parse_args()
    run_plot_qm9(args)
