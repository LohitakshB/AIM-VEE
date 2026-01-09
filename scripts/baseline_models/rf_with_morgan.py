"""Train a random forest regressor on Morgan fingerprints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "scikit-learn is required for random forest regression. Install it and retry."
    ) from exc

repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "pyproject.toml").exists():
    repo_root = repo_root.parent
src_root = repo_root / "src"
if src_root.exists() and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from aimvee.qm9_utils.train_utils import (  # noqa: E402
    build_rf_morgan_parser,
    dump_model,
    load_morgan_dataset,
    parse_max_features,
)
from aimvee.utils import ensure_dir  # noqa: E402


def main() -> None:
    parser = build_rf_morgan_parser()
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("Loading datasets...")
    x_train, y_train = load_morgan_dataset(train_csv, args.radius, args.n_bits)
    x_val, y_val = load_morgan_dataset(val_csv, args.radius, args.n_bits)
    print(f"Train size: {x_train.shape[0]} | Val size: {x_val.shape[0]}")

    print("Training random forest...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=parse_max_features(args.max_features),
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    val_preds = model.predict(x_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    print(f"val_mae={val_mae:.6f}")

    model_path = output_dir / "rf_morgan.pkl"
    dump_model(model, model_path)
    metrics = {"val_mae": float(val_mae)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
