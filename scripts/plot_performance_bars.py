#!/usr/bin/env python3
"""Generate MAE and RMSE bar charts with x-axis as model and colors by split."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Standalone configuration.
BASE_DIR = Path("outputs")
OUT_PATH = Path("outputs/performance_bar_mae.png")
SPLIT_ORDER = ["random", "size", "bemis-murcko", "tail-split"]
MODEL_ORDER = ["chemprop", "rf_morgan", "schnet", "mff_mlp", "umff-mlp "]

_PRIMARY_COLOR = "#0072B2"
_SECONDARY_COLOR = "#2b2b2b"
_SPLIT_COLORS = {
    "random": "#0072B2",
    "size": "#009E73",
    "bemis-murcko": "#E69F00",
    "tail-split": "#D55E00",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})


def _draw_grouped_bars(ax, x, width, groups, values, color_map):
    for i, group in enumerate(groups):
        offset = (i - (len(groups) - 1) / 2) * width
        ax.bar(
            x + offset,
            values[i],
            width=width,
            label=group,
            color=color_map.get(group, _PRIMARY_COLOR),
            edgecolor="black",
            linewidth=0.4,
        )


def _read_metrics(base_dir: Path):
    rows = []
    for path in sorted(base_dir.glob("*_eval/*/metrics_summary.json")):
        model = path.parts[-3].replace("_eval", "")
        split = path.parts[-2]
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_metrics = payload.get("model", {})
        mae = model_metrics.get("mae")
        rmse = model_metrics.get("rmse")
        if mae is None or rmse is None:
            continue
        rows.append(
            {
                "model": model,
                "split": split,
                "mae": float(mae),
                "rmse": float(rmse),
            }
        )
    return rows


def _ordered_unique(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pivot(rows, row_keys, col_keys, metric):
    matrix = np.full((len(row_keys), len(col_keys)), np.nan, dtype=float)
    row_idx = {name: i for i, name in enumerate(row_keys)}
    col_idx = {name: i for i, name in enumerate(col_keys)}
    for row in rows:
        matrix[row_idx[row["split"]], col_idx[row["model"]]] = row[metric]
    return matrix


def _plot_metric(
    out_path: Path,
    metric_matrix: np.ndarray,
    metric_label: str,
    splits: list[str],
    models: list[str],
) -> None:
    x = np.arange(len(models))
    width = 0.75 / len(splits)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _draw_grouped_bars(ax, x, width, splits, metric_matrix, _SPLIT_COLORS)

    finite = metric_matrix[np.isfinite(metric_matrix)]
    ymax = float(np.max(finite)) if finite.size else 1.0
    ax.set_ylim(0, ymax * 1.08)

    model_labels = [m.replace("_", " ").title() for m in models]
    split_labels = [s.replace("-", " ").title() for s in splits]
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_xlabel("Model", color=_SECONDARY_COLOR)
    ax.set_ylabel(metric_label, color=_SECONDARY_COLOR)
    ax.tick_params(axis="both", colors=_SECONDARY_COLOR)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        split_labels,
        loc="upper center",
        ncol=min(len(splits), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart to {out_path}")


def main() -> None:
    rows = _read_metrics(BASE_DIR)
    if not rows:
        raise SystemExit(f"No metrics_summary.json files found in {BASE_DIR}")

    available_splits = _ordered_unique(row["split"] for row in rows)
    available_models = _ordered_unique(row["model"] for row in rows)

    splits = [s for s in SPLIT_ORDER if s in available_splits] + [
        s for s in available_splits if s not in SPLIT_ORDER
    ]
    models = [m for m in MODEL_ORDER if m in available_models] + [
        m for m in available_models if m not in MODEL_ORDER
    ]

    mae = _pivot(rows, splits, models, "mae")
    rmse = _pivot(rows, splits, models, "rmse")

    rmse_out = OUT_PATH.with_name(
        f"{OUT_PATH.stem.replace('mae', 'rmse')}{OUT_PATH.suffix}"
    )
    if rmse_out == OUT_PATH:
        rmse_out = OUT_PATH.with_name(f"{OUT_PATH.stem}_rmse{OUT_PATH.suffix}")

    _plot_metric(OUT_PATH, mae, "MAE (eV)", splits, models)
    _plot_metric(rmse_out, rmse, "RMSE (eV)", splits, models)


if __name__ == "__main__":
    main()
