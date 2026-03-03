"""Run model inference on an input CSV and generate evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, global_mean_pool
from torch.utils.data import DataLoader as TorchDataLoader

from aimvee.datasets.geometry import GeometryCsvDataset, GeometryQemfiDataset
from aimvee.datasets.qemfi import QemfiDataset
from aimvee.experiments.mff_mlp import (
    _build_schnet_model as _build_mff_schnet,
    _capture_schnet_latent as _capture_latent,
    _load_geometry_rows,
    _load_qemfi_assets,
    _load_qemfi_model,
    _pick_schnet_latent_layer,
    _predict_qemfi_energies,
)
from aimvee.experiments.umff_mlp import _ensemble_predictions
from aimvee.features.coulomb import build_cm_reps
from aimvee.features.morgan import load_morgan_dataset
from aimvee.models import MFFMLP
from aimvee.utils import ensure_dir, select_device

matplotlib.use("Agg")


_STYLE = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.4,
}

_PRIMARY_COLOR = "#1f77b4"
_SECONDARY_COLOR = "#2b2b2b"
CM_SIZE = 22


def _apply_style() -> None:
    matplotlib.rcParams.update(_STYLE)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    rho, _ = spearmanr(x, y)
    if rho is None or np.isnan(rho):
        return float("nan")
    return float(rho)


def _metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    mae = mean_absolute_error(targets, preds)
    rmse = math.sqrt(mean_squared_error(targets, preds))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2_score(targets, preds)),
        "spearman": _spearmanr(targets, preds),
    }


def _uq_metrics(
    targets: np.ndarray, preds: np.ndarray, stds: np.ndarray
) -> Dict[str, float]:
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    stds = np.asarray(stds).reshape(-1)
    residuals = targets - preds
    abs_err = np.abs(residuals)
    sigma = np.maximum(stds, 1e-12)
    nll = float(
        np.mean(
            0.5 * np.log(2 * math.pi * sigma**2) + 0.5 * (residuals**2) / (sigma**2)
        )
    )
    return {
        "uq_rank_corr": float(_spearmanr(abs_err, stds)),
        "uq_nll": nll,
        "uq_sharpness": float(np.mean(stds)),
        "uq_dispersion_cv": float(np.std(stds) / max(np.mean(stds), 1e-12)),
    }


def _plot_parity(
    targets: np.ndarray,
    preds: np.ndarray,
    path: Path,
    xlim: Optional[Tuple[float, float]],
) -> None:
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(targets, preds, s=10, alpha=0.7, color=_PRIMARY_COLOR)
    min_val = float(min(targets.min(), preds.min()))
    max_val = float(max(targets.max(), preds.max()))
    if xlim is not None:
        min_val, max_val = xlim
    plt.plot([min_val, max_val], [min_val, max_val], color=_SECONDARY_COLOR, linewidth=1)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Parity Plot")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_residuals(
    targets: np.ndarray,
    preds: np.ndarray,
    path: Path,
    bins: int,
    xlim: Optional[Tuple[float, float]],
) -> None:
    residuals = np.asarray(targets).reshape(-1) - np.asarray(preds).reshape(-1)
    plt.figure(figsize=(4.8, 3.8))
    plt.hist(
        residuals,
        bins=bins,
        range=xlim,
        alpha=0.8,
        color=_PRIMARY_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    plt.xlabel("Residual (target - pred)")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_abs_error(
    targets: np.ndarray,
    preds: np.ndarray,
    path: Path,
    bins: int,
    xlim: Optional[Tuple[float, float]],
) -> None:
    abs_err = np.abs(np.asarray(targets).reshape(-1) - np.asarray(preds).reshape(-1))
    plt.figure(figsize=(4.8, 3.8))
    plt.hist(
        abs_err,
        bins=bins,
        range=xlim,
        alpha=0.8,
        color=_PRIMARY_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.title("Absolute Error Histogram")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_mae_rmse_bar(targets: np.ndarray, preds: np.ndarray, path: Path) -> None:
    err = np.asarray(preds).reshape(-1) - np.asarray(targets).reshape(-1)
    mae_total = float(np.mean(np.abs(err)))
    rmse_total = float(np.sqrt(np.mean(err**2)))

    plt.figure(figsize=(5.5, 4.5))
    plt.bar(
        ["MAE", "RMSE"],
        [mae_total, rmse_total],
        color=[_PRIMARY_COLOR, "#E53935"],
        alpha=0.85,
        width=0.55,
    )
    plt.ylabel("Error (eV)")
    plt.title("Overall MAE and RMSE")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_uncertainty_hist(
    stds: np.ndarray, path: Path, bins: int, xlim: Optional[Tuple[float, float]]
) -> None:
    stds = np.asarray(stds).reshape(-1)
    plt.figure(figsize=(4.8, 3.8))
    plt.hist(
        stds,
        bins=bins,
        range=xlim,
        alpha=0.8,
        color=_PRIMARY_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    plt.xlabel("Predicted Std")
    plt.ylabel("Count")
    plt.title("Uncertainty Histogram")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_error_vs_uncertainty_loglog(
    targets: np.ndarray, preds: np.ndarray, stds: np.ndarray, path: Path
) -> None:
    abs_err = np.abs(np.asarray(targets).reshape(-1) - np.asarray(preds).reshape(-1))
    stds = np.asarray(stds).reshape(-1)
    eps = 1e-8
    abs_err = np.maximum(abs_err, eps)
    stds = np.maximum(stds, eps)

    plt.figure(figsize=(4.8, 4.2))
    plt.scatter(abs_err, stds, s=10, alpha=0.6, color=_PRIMARY_COLOR)

    min_val = float(min(abs_err.min(), stds.min()))
    max_val = float(max(abs_err.max(), stds.max()))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color=_SECONDARY_COLOR,
        linewidth=1,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Absolute Error (|target - pred|)")
    plt.ylabel("Predicted Uncertainty (Std)")
    plt.title("Error vs Predicted Uncertainty (Log-Log)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_parity_with_uncertainty_bars(
    targets: np.ndarray,
    preds: np.ndarray,
    stds: np.ndarray,
    path: Path,
    xlim: Optional[Tuple[float, float]],
) -> None:
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    stds = np.asarray(stds).reshape(-1)

    # Reduce visual clutter on large sets while preserving the full value range.
    max_points = 1600
    if targets.size > max_points:
        idx = np.linspace(0, targets.size - 1, max_points, dtype=int)
        targets = targets[idx]
        preds = preds[idx]
        stds = stds[idx]

    plt.figure(figsize=(4.8, 4.8))
    plt.errorbar(
        targets,
        preds,
        yerr=stds,
        fmt="none",
        ecolor=_PRIMARY_COLOR,
        alpha=0.28,
        elinewidth=1.2,
        capsize=0,
    )
    plt.scatter(targets, preds, s=9, alpha=0.65, color=_PRIMARY_COLOR)

    min_val = float(min(targets.min(), preds.min()))
    max_val = float(max(targets.max(), preds.max()))
    if xlim is not None:
        min_val, max_val = xlim
    plt.plot([min_val, max_val], [min_val, max_val], color=_SECONDARY_COLOR, linewidth=1)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Parity Plot with Uncertainty Bars")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_calibration(
    targets: np.ndarray, preds: np.ndarray, stds: np.ndarray, path: Path
) -> None:
    residuals = targets - preds
    quantiles = np.linspace(0.05, 0.95, 19)
    z_scores = norm.ppf(quantiles)
    observed = []
    for z in z_scores:
        observed.append(float(np.mean(residuals <= z * stds)))
    plt.figure(figsize=(4.8, 3.8))
    plt.plot(quantiles, observed, marker="o", color=_PRIMARY_COLOR)
    plt.plot([0, 1], [0, 1], linestyle="--", color=_SECONDARY_COLOR, linewidth=1)
    plt.xlabel("Expected CDF")
    plt.ylabel("Observed CDF")
    plt.title("Calibration Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_summary(path: Path, metrics: Dict[str, float]) -> None:
    summary = {"model": metrics}
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _ensure_target_column(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "target" not in reader.fieldnames:
            raise ValueError("Input CSV must include a 'target' column for metrics.")


def _load_targets_from_input(path: Path) -> np.ndarray:
    targets = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            val = row.get("target", "").strip()
            if val != "":
                targets.append(float(val))
    if not targets:
        raise ValueError("Input CSV contains no target values.")
    return np.asarray(targets, dtype=float)


def _resolve_chemprop_predict_cmd() -> List[str]:
    cmd = shutil.which("chemprop")
    if cmd:
        return [cmd, "predict"]
    cmd = shutil.which("chemprop_predict")
    if cmd:
        return [cmd]
    return []


def _parse_prediction_csv(
    path: Path, *, fallback_target_as_pred: bool = False
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Prediction CSV missing header.")
        fields = {name.lower(): name for name in reader.fieldnames}
        target_col = fields.get("target") or fields.get("target_0")
        pred_col = (
            fields.get("pred_mean")
            or fields.get("pred_0")
            or fields.get("preds")
            or fields.get("prediction")
        )
        if pred_col is None and fallback_target_as_pred and target_col is not None:
            pred_col = target_col
            target_col = None
        if pred_col is None:
            raise ValueError("Prediction CSV must include a prediction column.")
        targets = []
        preds = []
        for row in reader:
            p_val = row.get(pred_col, "").strip()
            if p_val == "":
                continue
            if target_col is not None:
                t_val = row.get(target_col, "").strip()
                if t_val != "":
                    targets.append(float(t_val))
            preds.append(float(p_val))
    if not targets:
        return None, np.asarray(preds, dtype=float)
    return np.asarray(targets, dtype=float), np.asarray(preds, dtype=float)


def _write_smiles_only_csv(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        smiles_column = "smiles"
        if reader.fieldnames is None or smiles_column not in reader.fieldnames:
            raise ValueError(
                f"Input CSV missing smiles column '{smiles_column}'."
            )
        with output_csv.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.writer(out_handle)
            writer.writerow([smiles_column])
            for row in reader:
                writer.writerow([row.get(smiles_column, "").strip()])


def _eval_schnet(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    dataset = GeometryCsvDataset(Path(args.input_csv))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = select_device(args.device)
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
    model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pt", map_location=device))
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch.z, batch.pos, batch.batch).cpu())
            targets.append(batch.y.view(-1).cpu())
    return torch.cat(targets, dim=0).numpy(), torch.cat(preds, dim=0).numpy()


def _eval_rf(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    model_path = Path(args.model_dir) / "rf_morgan.pkl"
    model = joblib.load(model_path)
    x, y = load_morgan_dataset(Path(args.input_csv), args.radius, args.n_bits)
    preds = model.predict(x)
    return y, preds


def _eval_chemprop(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    cmd = _resolve_chemprop_predict_cmd()
    if not cmd:
        raise SystemExit(
            "Chemprop CLI not found. Install chemprop and ensure it is on PATH."
        )
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    preds_path = output_dir / "chemprop_preds.csv"
    extra_args = shlex.split(args.chemprop_args) if args.chemprop_args else []
    if args.chemprop_checkpoint_path:
        checkpoint_arg = ["--checkpoint_path", str(args.chemprop_checkpoint_path)]
    else:
        checkpoint_dir = (
            Path(args.chemprop_checkpoint_dir)
            if args.chemprop_checkpoint_dir
            else Path(args.model_dir)
        )
        checkpoint_arg = ["--checkpoint_dir", str(checkpoint_dir)]

    def _run_predict(test_path: Path) -> None:
        predict_cmd = [
            *cmd,
            "--test_path",
            str(test_path),
            *checkpoint_arg,
            "--preds_path",
            str(preds_path),
            "--smiles_columns",
            "smiles",
        ]
        predict_cmd.extend(extra_args)
        subprocess.run(predict_cmd, check=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Always use a smiles-only CSV to avoid Chemprop parsing non-smiles columns.
    smiles_only = output_dir / "chemprop_smiles_only.csv"
    _write_smiles_only_csv(input_csv, smiles_only)
    _run_predict(smiles_only)
    input_dir = Path(args.input_csv).parent
    candidates = [
        preds_path,
        output_dir / "preds.csv",
        output_dir / "predictions.csv",
        output_dir / "test_preds.csv",
        output_dir / "test_predictions.csv",
        Path(args.model_dir) / "preds.csv",
        Path(args.model_dir) / "predictions.csv",
        Path(args.model_dir) / "test_preds.csv",
        Path(args.model_dir) / "test_predictions.csv",
        input_dir / "preds.csv",
        input_dir / "predictions.csv",
        input_dir / "test_preds.csv",
        input_dir / "test_predictions.csv",
        Path("preds.csv"),
        Path("predictions.csv"),
        Path("test_preds.csv"),
        Path("test_predictions.csv"),
    ]
    found_path = next((path for path in candidates if path.exists()), None)
    if found_path is None:
        raise FileNotFoundError(
            "Chemprop did not produce a predictions CSV. "
            "Try adding '--chemprop-args \"--preds_path <path>\"' "
            "to control where Chemprop writes predictions. "
            f"Tried: {[str(p) for p in candidates]}"
        )
    targets, preds = _parse_prediction_csv(found_path, fallback_target_as_pred=True)
    if preds is None or preds.size == 0:
        raise ValueError("Chemprop predictions file is empty.")

    if targets is None:
        # Chemprop predict output lacks targets; load from input CSV.
        targets = _load_targets_from_input(Path(args.input_csv))

    if found_path == preds_path:
        preds_path.unlink(missing_ok=True)
    return targets, preds


def _eval_mff(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    device = select_device(args.device)
    model_dir = Path(args.model_dir)
    model_path, scaler_x, scaler_y, pca, _ = _load_qemfi_assets(Path(args.qemfi_model_dir))
    qemfi_model, model_d_rep, n_fids, n_states = _load_qemfi_model(model_path, device)
    if args.qemfi_fidelity < 0 or args.qemfi_fidelity >= n_fids:
        raise ValueError("qemfi-fidelity out of range.")
    rows = _load_geometry_rows(Path(args.input_csv))
    cm = build_cm_reps(rows, CM_SIZE)
    qemfi = _predict_qemfi_energies(
        qemfi_model,
        cm,
        scaler_x,
        scaler_y,
        pca,
        args.qemfi_fidelity,
        n_states,
        args.n_lowest_states,
        args.qemfi_batch_size,
        device,
    )
    scaler_path = model_dir / "qemfi_scaler.pkl"
    if scaler_path.exists():
        qemfi_scaler = joblib.load(scaler_path)
        qemfi = qemfi_scaler.transform(qemfi)
    dataset = GeometryQemfiDataset(rows, qemfi)
    loader = DataLoader(dataset, batch_size=args.schnet_batch_size, shuffle=False)
    schnet_model = _build_mff_schnet(args, device)
    latent_name, latent_layer, latent_dim = _pick_schnet_latent_layer(schnet_model)
    _ = latent_name, latent_dim
    mff_mlp = MFFMLP(
        in_dim=qemfi.shape[1] + latent_dim,
        hidden_dim=args.mlp_hidden_dim,
        num_layers=args.mlp_layers,
        dropout=args.mlp_dropout,
    ).to(device)
    schnet_model.load_state_dict(torch.load(model_dir / "best_schnet.pt", map_location=device))
    mff_mlp.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    latent_handle, hook_handle = _capture_latent(latent_layer)
    schnet_model.eval()
    mff_mlp.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
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
            qemfi_batch = batch.qemfi
            if qemfi_batch.dim() == 1:
                qemfi_batch = qemfi_batch.unsqueeze(1)
            features = torch.cat([qemfi_batch, latent], dim=1)
            preds.append(mff_mlp(features).cpu())
            targets.append(batch.y.view(-1).cpu())
    hook_handle.remove()
    return torch.cat(targets, dim=0).numpy(), torch.cat(preds, dim=0).numpy()


def _eval_umff(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = select_device(args.device)
    model_dir = Path(args.model_dir)
    model_path, scaler_x, scaler_y, pca, _ = _load_qemfi_assets(Path(args.qemfi_model_dir))
    qemfi_model, model_d_rep, n_fids, n_states = _load_qemfi_model(model_path, device)
    if args.qemfi_fidelity < 0 or args.qemfi_fidelity >= n_fids:
        raise ValueError("qemfi-fidelity out of range.")
    rows = _load_geometry_rows(Path(args.input_csv))
    cm = build_cm_reps(rows, CM_SIZE)
    qemfi = _predict_qemfi_energies(
        qemfi_model,
        cm,
        scaler_x,
        scaler_y,
        pca,
        args.qemfi_fidelity,
        n_states,
        args.n_lowest_states,
        args.qemfi_batch_size,
        device,
    )
    scaler_path = model_dir / "qemfi_scaler.pkl"
    if scaler_path.exists():
        qemfi_scaler = joblib.load(scaler_path)
        qemfi = qemfi_scaler.transform(qemfi)
    dataset = GeometryQemfiDataset(rows, qemfi)
    loader = DataLoader(dataset, batch_size=args.schnet_batch_size, shuffle=False)
    ensemble = []
    latent_handles = []
    hook_handles = []
    member_dirs = sorted(model_dir.glob("ensemble_*"))
    if not member_dirs:
        raise FileNotFoundError(f"No ensemble_* directories found in {model_dir}")
    for member_dir in member_dirs:
        schnet_model = _build_mff_schnet(args, device)
        _, latent_layer, latent_dim = _pick_schnet_latent_layer(schnet_model)
        mff_mlp = MFFMLP(
            in_dim=qemfi.shape[1] + latent_dim,
            hidden_dim=args.mlp_hidden_dim,
            num_layers=args.mlp_layers,
            dropout=args.mlp_dropout,
        ).to(device)
        schnet_model.load_state_dict(
            torch.load(member_dir / "best_schnet.pt", map_location=device)
        )
        mff_mlp.load_state_dict(
            torch.load(member_dir / "best_model.pt", map_location=device)
        )
        latent_handle, hook_handle = _capture_latent(latent_layer)
        ensemble.append((schnet_model, mff_mlp))
        latent_handles.append(latent_handle)
        hook_handles.append(hook_handle)
    targets, mean_preds, std_preds = _ensemble_predictions(
        ensemble, latent_handles, loader, device
    )
    for handle in hook_handles:
        handle.remove()
    return targets, mean_preds, std_preds


def _eval_qemfi(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if not args.qemfi_data_dir:
        raise ValueError("--qemfi-data-dir is required for qemfi.")

    data_dir = Path(args.qemfi_data_dir)
    method = args.qemfi_method
    x_test = np.load(data_dir / f"{method}_test.npy")
    y_test = np.load(data_dir / "EV_test.npy")

    mask = y_test >= 0
    x_test = x_test[mask]
    y_test = y_test[mask]
    if x_test.shape[0] == 0:
        raise ValueError("QeMFi test set is empty after filtering invalid targets.")

    model_path, scaler_x, scaler_y, pca, _ = _load_qemfi_assets(Path(args.model_dir))
    device = select_device(args.device)
    model, model_d_rep, _, _ = _load_qemfi_model(model_path, device)

    d_rep_raw = x_test.shape[1] - 2
    x_test_rep = x_test[:, :d_rep_raw].astype(np.float32)
    x_test_idx = x_test[:, d_rep_raw:].astype(np.int64)

    x_test_rep = scaler_x.transform(x_test_rep)
    if pca is not None:
        x_test_rep = pca.transform(x_test_rep)

    if x_test_rep.shape[1] != model_d_rep:
        raise ValueError(
            "QeMFi representation dimension mismatch between prepared arrays and model."
        )

    x_test_proc = np.concatenate(
        [x_test_rep.astype(np.float32), x_test_idx.astype(np.int64)], axis=1
    )
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    test_ds = QemfiDataset(x_test_proc, y_test_scaled)
    test_loader = TorchDataLoader(
        test_ds, batch_size=args.qemfi_eval_batch_size, shuffle=False
    )

    preds_scaled: List[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(
                batch["feats"].to(device),
                batch["fid_id"].to(device),
                batch["state_id"].to(device),
            )
            preds_scaled.append(pred.detach().cpu().numpy())

    preds_scaled_arr = np.concatenate(preds_scaled, axis=0).reshape(-1, 1)
    preds_cm = scaler_y.inverse_transform(preds_scaled_arr).ravel()
    targets_cm = y_test.reshape(-1)

    cm_to_ev = 1.239841984e-4
    return targets_cm * cm_to_ev, preds_cm * cm_to_ev


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on an input CSV and generate graphs.",
        add_help=add_help,
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["qemfi", "schnet", "rf_morgan", "chemprop", "mff_mlp", "umff_mlp"],
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot-bins", type=int, default=50)
    parser.add_argument("--plot-parity-range", type=str, default="")
    parser.add_argument("--plot-residual-range", type=str, default="")
    parser.add_argument("--plot-abs-error-range", type=str, default="")
    parser.add_argument("--plot-uncertainty-range", type=str, default="")

    # SchNet hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--max-num-neighbors", type=int, default=32)
    parser.add_argument("--readout", choices=("add", "mean"), default="add")
    parser.add_argument("--dipole", action="store_true")

    # Morgan RF
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--chemprop-args", default=None)
    parser.add_argument("--chemprop-target-column", default="target")
    parser.add_argument("--chemprop-checkpoint-dir", default=None)
    parser.add_argument("--chemprop-checkpoint-path", default=None)

    # QeMFi / MFF / UMFF
    parser.add_argument("--schnet-hidden-dim", type=int, default=128)
    parser.add_argument("--schnet-num-filters", type=int, default=128)
    parser.add_argument("--schnet-num-interactions", type=int, default=6)
    parser.add_argument("--schnet-num-gaussians", type=int, default=50)
    parser.add_argument("--schnet-cutoff", type=float, default=10.0)
    parser.add_argument("--schnet-max-num-neighbors", type=int, default=32)
    parser.add_argument("--schnet-readout", choices=("add", "mean"), default="add")
    parser.add_argument("--schnet-dipole", action="store_true")
    parser.add_argument("--qemfi-model-dir")
    parser.add_argument("--qemfi-fidelity", type=int, default=4)
    parser.add_argument("--qemfi-batch-size", type=int, default=2048)
    parser.add_argument("--qemfi-data-dir", default="")
    parser.add_argument("--qemfi-method", default="CM")
    parser.add_argument("--qemfi-eval-batch-size", type=int, default=256)
    parser.add_argument("--n-lowest-states", type=int, default=10)
    parser.add_argument("--schnet-batch-size", type=int, default=32)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)

    return parser


def run_evaluate_model(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)

    _apply_style()
    if args.model_type != "qemfi":
        if not args.input_csv:
            raise ValueError("--input-csv is required for this model type.")
        input_csv = Path(args.input_csv)
        _ensure_target_column(input_csv)

    if args.model_type == "qemfi":
        targets, preds = _eval_qemfi(args)
        stds = None
    elif args.model_type == "schnet":
        targets, preds = _eval_schnet(args)
        stds = None
    elif args.model_type == "rf_morgan":
        targets, preds = _eval_rf(args)
        stds = None
    elif args.model_type == "chemprop":
        targets, preds = _eval_chemprop(args)
        stds = None
    elif args.model_type == "mff_mlp":
        if not args.qemfi_model_dir:
            raise ValueError("--qemfi-model-dir is required for mff_mlp.")
        targets, preds = _eval_mff(args)
        stds = None
    elif args.model_type == "umff_mlp":
        if not args.qemfi_model_dir:
            raise ValueError("--qemfi-model-dir is required for umff_mlp.")
        targets, preds, stds = _eval_umff(args)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    metrics = _metrics(targets, preds)
    if stds is not None:
        metrics.update(_uq_metrics(targets, preds, stds))

    _write_summary(output_dir / "metrics_summary.json", metrics)

    def _parse_range(value: str) -> Optional[Tuple[float, float]]:
        if not value:
            return None
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError("Plot ranges must be formatted as 'min,max'.")
        return float(parts[0]), float(parts[1])

    parity_range = _parse_range(args.plot_parity_range)
    residual_range = _parse_range(args.plot_residual_range)
    abs_error_range = _parse_range(args.plot_abs_error_range)
    uncertainty_range = _parse_range(args.plot_uncertainty_range)

    _plot_parity(targets, preds, plots_dir / "parity.png", parity_range)
    _plot_residuals(
        targets, preds, plots_dir / "residuals.png", args.plot_bins, residual_range
    )
    _plot_abs_error(
        targets, preds, plots_dir / "abs_error.png", args.plot_bins, abs_error_range
    )
    if args.model_type == "qemfi":
        _plot_mae_rmse_bar(targets, preds, plots_dir / "mae_rmse_bar.png")
    if stds is not None:
        _plot_uncertainty_hist(
            stds, plots_dir / "uncertainty.png", args.plot_bins, uncertainty_range
        )
        _plot_parity_with_uncertainty_bars(
            targets,
            preds,
            stds,
            plots_dir / "parity_uncertainty_bars.png",
            parity_range,
        )
        _plot_error_vs_uncertainty_loglog(
            targets, preds, stds, plots_dir / "error_vs_uncertainty_loglog.png"
        )
        _plot_calibration(targets, preds, stds, plots_dir / "calibration.png")

    print(f"Saved metrics summary to {output_dir / 'metrics_summary.json'}")
    print(f"Saved plots to {plots_dir}")


def main() -> None:
    args = build_parser().parse_args()
    run_evaluate_model(args)


if __name__ == "__main__":
    main()
