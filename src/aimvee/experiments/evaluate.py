"""Evaluation suite for model predictions and uncertainty metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import f, norm, studentized_range
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class PredictionBatch:
    targets: np.ndarray
    preds: np.ndarray
    stds: Optional[np.ndarray] = None
    stds_calibrated: Optional[np.ndarray] = None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = values[order]

    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size != y.size:
        raise ValueError("Spearman inputs must be the same length.")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    cov = np.mean((rx - rx_mean) * (ry - ry_mean))
    std_x = rx.std()
    std_y = ry.std()
    if std_x == 0 or std_y == 0:
        return float("nan")
    return float(cov / (std_x * std_y))


def _calibration_rmse(
    residuals: np.ndarray, stds: np.ndarray, quantiles: np.ndarray
) -> Tuple[float, float]:
    if residuals.size == 0:
        return float("nan"), float("nan")
    z_scores = norm.ppf(quantiles)
    observed = []
    for z in z_scores:
        observed.append(float(np.mean(residuals <= z * stds)))
    observed_arr = np.asarray(observed)
    diff = observed_arr - quantiles
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def _nll(residuals: np.ndarray, stds: np.ndarray) -> float:
    eps = 1e-12
    sigma2 = np.maximum(stds, eps) ** 2
    return float(
        np.mean(0.5 * np.log(2 * math.pi * sigma2) + 0.5 * (residuals**2) / sigma2)
    )


def _uq_metrics(
    targets: np.ndarray, preds: np.ndarray, stds: np.ndarray
) -> Dict[str, float]:
    residuals = targets - preds
    abs_err = np.abs(residuals)
    rmse_cal, mae_cal = _calibration_rmse(residuals, stds, _DEFAULT_QUANTILES)
    return {
        "uq_rank_corr": _spearmanr(abs_err, stds),
        "uq_nll": _nll(residuals, stds),
        "uq_sharpness": float(np.mean(stds)),
        "uq_dispersion_cv": float(np.std(stds) / max(np.mean(stds), 1e-12)),
        "uq_calibration_rmse": rmse_cal,
        "uq_calibration_mae": mae_cal,
    }


def _regression_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(targets, preds)
    rmse = math.sqrt(mean_squared_error(targets, preds))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2_score(targets, preds)),
        "spearman": _spearmanr(targets, preds),
    }


def _parse_predictions(path: Path) -> PredictionBatch:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in predictions file {path}")
        columns = {name.strip().lower(): name for name in reader.fieldnames}

        def pick(candidates: Iterable[str]) -> Optional[str]:
            for candidate in candidates:
                key = candidate.lower()
                if key in columns:
                    return columns[key]
            return None

        target_col = pick(
            ["target", "target_0", "y", "y_true", "label"]
        )
        pred_col = pick(
            ["prediction", "pred", "pred_0", "pred_mean", "y_pred", "preds"]
        )
        std_col = pick(["pred_std", "std", "sigma", "pred_sigma"])
        std_cal_col = pick([
            "pred_std_calibrated",
            "std_calibrated",
            "pred_std_cal",
        ])

        if target_col is None or pred_col is None:
            raise ValueError(
                f"Predictions file {path} must include target and prediction columns."
            )

        targets: List[float] = []
        preds: List[float] = []
        stds: List[float] = []
        stds_cal: List[float] = []

        for row in reader:
            target_val = row.get(target_col, "").strip()
            pred_val = row.get(pred_col, "").strip()
            if target_val == "" or pred_val == "":
                continue
            targets.append(float(target_val))
            preds.append(float(pred_val))
            if std_col:
                std_val = row.get(std_col, "").strip()
                stds.append(float(std_val)) if std_val != "" else stds.append(float("nan"))
            if std_cal_col:
                std_cal_val = row.get(std_cal_col, "").strip()
                stds_cal.append(
                    float(std_cal_val) if std_cal_val != "" else float("nan")
                )

    if not targets:
        raise ValueError(f"No prediction rows found in {path}")

    stds_arr = np.asarray(stds, dtype=float) if stds else None
    stds_cal_arr = np.asarray(stds_cal, dtype=float) if stds_cal else None

    return PredictionBatch(
        targets=np.asarray(targets, dtype=float),
        preds=np.asarray(preds, dtype=float),
        stds=stds_arr,
        stds_calibrated=stds_cal_arr,
    )


def _discover_prediction_paths(
    results_root: Path,
    models: Optional[List[str]],
    prediction_map: Dict[str, str],
    default_pred_name: str,
    split_methods: Optional[List[str]],
) -> List[Tuple[str, str, Path]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    split_dirs = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    if split_methods:
        split_dirs = [path for path in split_dirs if path.name in split_methods]

    found: List[Tuple[str, str, Path]] = []
    for split_dir in split_dirs:
        model_dirs = [
            path
            for path in split_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        if models:
            model_dirs = [path for path in model_dirs if path.name in models]
        for model_dir in model_dirs:
            model_name = model_dir.name
            pred_name = prediction_map.get(model_name, default_pred_name)
            pred_path = Path(pred_name)
            if not pred_path.is_absolute():
                pred_path = model_dir / pred_name
            if pred_path.exists():
                found.append((split_dir.name, model_name, pred_path))
    if not found:
        raise ValueError("No prediction files found. Check results root and filenames.")
    return found


def _summarize_metrics(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        model = row["model"]
        for key, value in row.items():
            if key in {"model", "split", "n"}:
                continue
            by_model.setdefault(model, {}).setdefault(key, []).append(float(value))
    for model, metrics in by_model.items():
        summary[model] = {}
        for key, values in metrics.items():
            values_arr = np.asarray(values, dtype=float)
            summary[model][f"{key}_mean"] = float(np.mean(values_arr))
            summary[model][f"{key}_std"] = float(np.std(values_arr, ddof=1)) if len(values_arr) > 1 else 0.0
    return summary


def _anova_tukey(
    rows: List[Dict[str, float]], metric: str, alpha: float
) -> Optional[Dict[str, object]]:
    data: Dict[str, Dict[str, float]] = {}
    for row in rows:
        split = row["split"]
        model = row["model"]
        value = row.get(metric)
        if value is None or math.isnan(value):
            continue
        data.setdefault(split, {})[model] = float(value)

    splits = [split for split, values in data.items() if values]
    if not splits:
        return None

    models = sorted({model for values in data.values() for model in values})
    complete_splits = [
        split for split in splits if all(model in data[split] for model in models)
    ]
    if len(complete_splits) < 2 or len(models) < 2:
        return None

    matrix = np.asarray(
        [[data[split][model] for model in models] for split in complete_splits],
        dtype=float,
    )
    n_splits, n_models = matrix.shape
    grand_mean = matrix.mean()
    model_means = matrix.mean(axis=0)
    split_means = matrix.mean(axis=1)

    ss_model = n_splits * np.sum((model_means - grand_mean) ** 2)
    ss_split = n_models * np.sum((split_means - grand_mean) ** 2)
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_error = ss_total - ss_model - ss_split

    df_model = n_models - 1
    df_error = (n_models - 1) * (n_splits - 1)
    if df_error <= 0:
        return None

    ms_model = ss_model / df_model
    ms_error = ss_error / df_error
    f_stat = ms_model / ms_error if ms_error > 0 else float("inf")
    p_value = float(f.sf(f_stat, df_model, df_error))

    se = math.sqrt(ms_error / n_splits)
    q_crit = float(studentized_range.ppf(1 - alpha, n_models, df_error))
    hsd = q_crit * se

    comparisons: List[Dict[str, float | str | bool]] = []
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if j <= i:
                continue
            diff = model_means[i] - model_means[j]
            q_stat = abs(diff) / se if se > 0 else float("inf")
            p_comp = float(studentized_range.sf(q_stat, n_models, df_error))
            comparisons.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_diff": float(diff),
                    "ci_low": float(diff - hsd),
                    "ci_high": float(diff + hsd),
                    "p_value": p_comp,
                    "significant": bool(abs(diff) > hsd),
                }
            )

    return {
        "metric": metric,
        "models": models,
        "splits": complete_splits,
        "n_splits": n_splits,
        "anova": {
            "f_stat": float(f_stat),
            "p_value": p_value,
            "df_model": int(df_model),
            "df_error": int(df_error),
            "ms_error": float(ms_error),
        },
        "tukey_hsd": comparisons,
    }


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions with statistical comparisons.",
        add_help=add_help,
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model directory names to include.",
    )
    parser.add_argument(
        "--prediction-file",
        default="predictions.csv",
        help="Default predictions filename within each model directory.",
    )
    parser.add_argument(
        "--predictions",
        nargs="*",
        default=None,
        help="Override prediction filename per model as model=filename.",
    )
    parser.add_argument(
        "--split-methods",
        nargs="*",
        default=None,
        help="Optional list of split method directories to include.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for metrics summaries (defaults to results-root).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Tukey HSD confidence intervals.",
    )
    return parser


_DEFAULT_QUANTILES = np.linspace(0.05, 0.95, 19)


def run_evaluation(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_map: Dict[str, str] = {}
    if args.predictions:
        for spec in args.predictions:
            if "=" not in spec:
                raise ValueError(
                    "Prediction overrides must be formatted as model=filename."
                )
            model, filename = spec.split("=", 1)
            prediction_map[model.strip()] = filename.strip()

    discovered = _discover_prediction_paths(
        results_root,
        args.models,
        prediction_map,
        args.prediction_file,
        args.split_methods,
    )

    rows: List[Dict[str, float]] = []
    for split_name, model_name, pred_path in discovered:
        batch = _parse_predictions(pred_path)
        metrics = _regression_metrics(batch.targets, batch.preds)
        entry: Dict[str, float | str] = {
            "split": split_name,
            "model": model_name,
            "n": float(batch.targets.size),
        }
        entry.update(metrics)

        if batch.stds is not None:
            entry.update(_uq_metrics(batch.targets, batch.preds, batch.stds))
        if batch.stds_calibrated is not None:
            uq_cal = _uq_metrics(
                batch.targets, batch.preds, batch.stds_calibrated
            )
            entry.update({f"{k}_cal": v for k, v in uq_cal.items()})
        rows.append(entry)  # type: ignore[arg-type]

    if not rows:
        raise ValueError("No metrics were generated from the prediction files.")

    summary = _summarize_metrics(rows)
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved metrics summary to {summary_path}")


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
