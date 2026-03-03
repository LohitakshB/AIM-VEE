# Codebase structure

This repository is organized around reusable modules plus CLI entrypoints.

## Top-level

- `src/aimvee/`: core package
- `docs/`: user documentation
- `scripts/`: utility scripts for metadata generation and plotting
- `notebooks/`: exploratory notebooks
- `data/`: local datasets (not tracked)
- `models/`: trained model artifacts (not tracked)
- `outputs/`: evaluation/prediction outputs (not tracked)

## `src/aimvee` package

- `cli.py`: main CLI dispatcher (`aimvee <command>`)
- `api.py`: Python API wrappers for main workflows
- `experiments/`: runnable experiment/inference/evaluation modules
- `data_utils/`: split creation and metadata prep logic
- `datasets/`: dataset adapters and XYZ parsing
- `features/`: feature generation (Coulomb, Morgan)
- `models/`: model definitions (QeMFi, MFF-MLP)
- `trainers/`: shared training loops
- `qemfi_utils/`: QeMFi-specific preprocessing helpers
- `utils.py`: shared utility functions

## Key CLI-backed modules

- `experiments/qemfi.py`: `generate-cm`, `prep-qemfi`, `train-qemfi`
- `experiments/data_prep.py`: `prepare-splits`
- `experiments/schnet.py`: `train-schnet`
- `experiments/rf_morgan.py`: `train-rf-morgan`
- `experiments/chemprop.py`: `train-chemprop`
- `experiments/mff_mlp.py`: `train-mff-mlp`
- `experiments/umff_mlp.py`: `train-umff-mlp`
- `experiments/infer_model.py`: `infer-model`
- `experiments/evaluate_model.py`: `evaluate-model`
- `experiments/evaluate.py`: `evaluate`

## Script utilities

- `scripts/generate_metadata.py`: build metadata CSV from raw inputs
- `scripts/plot_performance_bars.py`: generate model performance bar plots
