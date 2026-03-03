"""Run model inference on an input CSV and write predictions."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np

from aimvee.experiments import evaluate_model as eval_model
from aimvee.utils import ensure_dir


def _chemprop_predict(
    args: argparse.Namespace,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    cmd = eval_model._resolve_chemprop_predict_cmd()  # noqa: SLF001
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
    smiles_only = output_dir / "chemprop_smiles_only.csv"
    eval_model._write_smiles_only_csv(input_csv, smiles_only)  # noqa: SLF001
    _run_predict(smiles_only)

    input_dir = input_csv.parent
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

    targets, preds = eval_model._parse_prediction_csv(found_path)  # noqa: SLF001
    if preds.size == 0:
        raise ValueError("Chemprop predictions file is empty.")

    if found_path == preds_path:
        preds_path.unlink(missing_ok=True)
    return targets, preds


def _infer_predictions(
    args: argparse.Namespace,
) -> tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if args.model_type == "schnet":
        targets, preds = eval_model._eval_schnet(args)  # noqa: SLF001
        return targets, preds, None, None
    if args.model_type == "rf_morgan":
        targets, preds = eval_model._eval_rf(args)  # noqa: SLF001
        return targets, preds, None, None
    if args.model_type == "chemprop":
        targets, preds = _chemprop_predict(args)
        return targets, preds, None, None
    if args.model_type == "mff_mlp":
        if not args.qemfi_model_dir:
            raise ValueError("--qemfi-model-dir is required for mff_mlp.")
        targets, preds = eval_model._eval_mff(args)  # noqa: SLF001
        return targets, preds, None, None
    if args.model_type == "umff_mlp":
        if not args.qemfi_model_dir:
            raise ValueError("--qemfi-model-dir is required for umff_mlp.")
        targets, preds, stds = eval_model._eval_umff(args)  # noqa: SLF001
        calibrated = None
        iso_path = Path(args.model_dir) / "isotonic_reg.pkl"
        if iso_path.exists():
            iso = joblib.load(iso_path)
            calibrated = np.asarray(iso.predict(stds), dtype=float)
        return targets, preds, stds, calibrated
    raise ValueError(f"Unsupported model type: {args.model_type}")


def _load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV missing header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _append_unique_columns(
    base_fields: Sequence[str],
    extra_fields: Sequence[str],
) -> list[str]:
    out = list(base_fields)
    for name in extra_fields:
        if name not in out:
            out.append(name)
    return out


def _write_predictions_csv(
    input_csv: Path,
    output_csv: Path,
    preds: np.ndarray,
    *,
    targets: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
    calibrated_stds: Optional[np.ndarray] = None,
) -> None:
    _, rows = _load_csv_rows(input_csv)
    n = len(rows)
    if preds.shape[0] != n:
        raise ValueError(
            f"Prediction count mismatch for {input_csv}: got {preds.shape[0]}, expected {n}."
        )

    preds = np.asarray(preds).reshape(-1)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "pred_value"])
        writer.writeheader()
        for idx, row in enumerate(rows):
            file_path = (
                row.get("xyz_path")
                or row.get("file_path")
                or row.get("path")
                or ""
            )
            writer.writerow(
                {
                    "file_path": file_path,
                    "pred_value": f"{float(preds[idx]):.12g}",
                }
            )


def _write_umff_predictions_csv(
    input_csv: Path,
    output_csv: Path,
    preds: np.ndarray,
    *,
    stds: Optional[np.ndarray],
    calibrated_stds: Optional[np.ndarray],
) -> None:
    _, rows = _load_csv_rows(input_csv)
    n = len(rows)

    preds = np.asarray(preds).reshape(-1)
    if preds.shape[0] != n:
        raise ValueError(
            f"Prediction count mismatch for {input_csv}: got {preds.shape[0]}, expected {n}."
        )

    if stds is None and calibrated_stds is None:
        raise ValueError("UMFF inference requires uncertainty values.")

    stds_arr = None if stds is None else np.asarray(stds).reshape(-1)
    calibrated_arr = (
        None if calibrated_stds is None else np.asarray(calibrated_stds).reshape(-1)
    )

    if stds_arr is not None and stds_arr.shape[0] != n:
        raise ValueError("Uncertainty length mismatch when writing UMFF predictions.")
    if calibrated_arr is not None and calibrated_arr.shape[0] != n:
        raise ValueError(
            "Calibrated uncertainty length mismatch when writing UMFF predictions."
        )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file_path", "pred_value", "pred_uncertainty"]
        )
        writer.writeheader()
        for idx, row in enumerate(rows):
            file_path = (
                row.get("xyz_path")
                or row.get("file_path")
                or row.get("path")
                or ""
            )
            uncertainty = (
                float(calibrated_arr[idx])
                if calibrated_arr is not None
                else float(stds_arr[idx])  # stds_arr is guaranteed non-None here.
            )
            writer.writerow(
                {
                    "file_path": file_path,
                    "pred_value": f"{float(preds[idx]):.12g}",
                    "pred_uncertainty": f"{uncertainty:.12g}",
                }
            )


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run model inference on an input CSV and write predictions "
            "without generating plots/metrics."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["schnet", "rf_morgan", "chemprop", "mff_mlp", "umff_mlp"],
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=32)

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
    parser.add_argument("--n-lowest-states", type=int, default=10)
    parser.add_argument("--schnet-batch-size", type=int, default=32)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)

    return parser


def run_infer_model(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    targets, preds, stds, calibrated_stds = _infer_predictions(args)

    predictions_csv = output_dir / "predictions.csv"
    if args.model_type == "umff_mlp":
        _write_umff_predictions_csv(
            Path(args.input_csv),
            predictions_csv,
            preds,
            stds=stds,
            calibrated_stds=calibrated_stds,
        )
    else:
        _write_predictions_csv(
            Path(args.input_csv),
            predictions_csv,
            preds,
            targets=targets,
            stds=stds,
            calibrated_stds=calibrated_stds,
        )

    print(f"Saved predictions to {predictions_csv}")


def main() -> None:
    args = build_parser().parse_args()
    run_infer_model(args)


if __name__ == "__main__":
    main()
