"""Random forest with Morgan fingerprints baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from aimvee.data_utils.data_prep import add_split_args, iter_split_csvs
from aimvee.features.morgan import load_morgan_dataset, parse_max_features
from aimvee.utils import ensure_dir


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a random forest regressor on Morgan fingerprints.",
        add_help=add_help,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", default="1.0")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--model-seed", type=int, default=13)
    return parser


def run_rf_morgan(args: argparse.Namespace) -> None:

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
        predefined_val_csv=(
            Path(args.predefined_val) if args.predefined_val else None
        ),
        predefined_test_csv=(
            Path(args.predefined_test) if args.predefined_test else None
        ),
    )

    for split_method, train_csv, val_csv, split_output in split_iter:
        print(f"Using splits ({split_method})...")
        output_dir = Path(args.output_dir) / split_method
        ensure_dir(output_dir)

        print("Loading datasets...")
        x_train, y_train = load_morgan_dataset(train_csv, args.radius, args.n_bits)
        x_val, y_val = load_morgan_dataset(val_csv, args.radius, args.n_bits)
        print(f"Train size: {x_train.shape[0]} | Val size: {x_val.shape[0]}")

        print("Training random forest...")
        model_seed = getattr(args, "model_seed", args.seed)
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=parse_max_features(args.max_features),
            n_jobs=args.n_jobs,
            random_state=model_seed,
        )
        model.fit(x_train, y_train)

        val_preds = model.predict(x_val)
        val_mae = float(np.mean(np.abs(y_val - val_preds)))
        print(f"val_mae={val_mae:.6f}")

        model_path = output_dir / "rf_morgan.pkl"
        try:
            import joblib  # type: ignore
        except Exception:
            import pickle

            with model_path.open("wb") as handle:
                pickle.dump(model, handle)
        else:
            joblib.dump(model, model_path)

        print(f"Saved model to {model_path}")

        test_csv = split_output / "test.csv"
        if test_csv.exists():
            x_test, y_test = load_morgan_dataset(
                test_csv, args.radius, args.n_bits
            )
            test_preds = model.predict(x_test)
            pred_path = output_dir / "test_predictions.csv"
            with pred_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["target", "pred_mean"])
                for target, pred in zip(y_test, test_preds):
                    writer.writerow([target, pred])



def main() -> None:
    parser = build_parser()
    add_split_args(parser, default_split_method=None)
    parser.set_defaults(id_column="geometry", target_column="output")
    args = parser.parse_args()
    run_rf_morgan(args)


if __name__ == "__main__":
    main()
