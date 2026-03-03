# AIM-VEE

AIM-VEE is a framework for predicting vertical excitation energies (VEE) with:
- a multi-fidelity fusion pipeline (QeMFi surrogate + SchNet latent + MFF-MLP / UMFF-MLP)
- baseline models (SchNet, Chemprop, RF Morgan)
- inference and evaluation tooling for predictions, plots, and aggregate metrics

## Installation

```bash
pip install .
```

## Data and model access (Zenodo)

- Datasets & Trained models are available on Zenodo: `TODO_ADD_ZENODO_MODELS_LINK`

## Quickstart

### 1) QeMFi surrogate pipeline

Generate Coulomb matrices:

```bash
python -m aimvee.cli generate-cm \
  --input-dir data/QeMFI_all/QeMFi \
  --output-dir data/QeMFI_all/QeMFi_cm
```

Prepare surrogate arrays:

```bash
python -m aimvee.cli prep-qemfi \
  --qemfi-dir data/QeMFI_all/QeMFi \
  --reps-dir data/QeMFI_all/QeMFi_cm \
  --data-dir data/QeMFI_all/model_training_test_data
```

Train surrogate:

```bash
python -m aimvee.cli train-qemfi \
  --data-dir data/QeMFI_all/model_training_test_data \
  --output-dir models/qemfi_surrogate
```

### 2) QM9-GWBSE splits

```bash
python -m aimvee.cli prepare-splits \
  --xyz-dir data/QM9GWBSE/QM9_xyz_files \
  --metadata data/QM9GWBSE/metadata.csv \
  --splits-output data/QM9GWBSE/splits
```

### 3) Train models

MFF-MLP:

```bash
python -m aimvee.cli train-mff-mlp \
  --xyz-dir data/QM9GWBSE/QM9_xyz_files \
  --metadata data/QM9GWBSE/metadata_e_exc_ss.csv \
  --splits-output data/QM9GWBSE/splits \
  --output-dir models/mff_mlp \
  --qemfi-model-dir models/qemfi_surrogate
```

UMFF-MLP:

```bash
python -m aimvee.cli train-umff-mlp \
  --xyz-dir data/QM9GWBSE/QM9_xyz_files \
  --metadata data/QM9GWBSE/metadata_e_exc_ss.csv \
  --splits-output data/QM9GWBSE/splits \
  --output-dir models/umff_mlp \
  --qemfi-model-dir models/qemfi_surrogate
```

Baselines:

```bash
python -m aimvee.cli train-schnet --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/schnet
python -m aimvee.cli train-rf-morgan --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/rf_morgan
python -m aimvee.cli train-chemprop --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/chemprop
```

### 4) Inference and evaluation

Predictions only (`predictions.csv`):

```bash
python -m aimvee.cli infer-model \
  --model-type mff_mlp \
  --model-dir models/mff_mlp/random \
  --qemfi-model-dir models/qemfi_surrogate \
  --input-csv data/QM9GWBSE/splits/random/test.csv \
  --output-dir outputs/mff_mlp_infer/random
```

Predictions + metrics + plots:

```bash
python -m aimvee.cli evaluate-model \
  --model-type mff_mlp \
  --model-dir models/mff_mlp/random \
  --qemfi-model-dir models/qemfi_surrogate \
  --input-csv data/QM9GWBSE/splits/random/test.csv \
  --output-dir outputs/mff_mlp_eval/random
```

Aggregate evaluation across multiple models/splits from saved prediction files:

```bash
python -m aimvee.cli evaluate \
  --results-root outputs \
  --output-json outputs/eval_summary.json
```

## Command index

Use `python -m aimvee.cli <command> --help` (or `aimvee <command> --help`).

- QeMFi utilities: `generate-cm`, `prep-qemfi`, `train-qemfi`
- Split prep: `prepare-splits`
- Training: `train-schnet`, `train-rf-morgan`, `train-chemprop`, `train-mff-mlp`, `train-umff-mlp`
- Inference/eval: `infer-model`, `evaluate-model`, `evaluate`

## Documentation map

- `docs/structure.md`: repository layout
- `docs/data.md`: expected data/artifact layout
- `docs/cli.md`: full CLI coverage
- `docs/workflows.md`: end-to-end workflows
- `docs/experiments.md`: experiment modules and outputs
- `docs/api.md`: Python API usage
- `docs/repro.md`: reproducibility checklist

## Notes

- `scripts/generate_metadata.py` can build the metadata CSV for QM9-GWBSE.
- Most training commands default to training across all split methods unless `--single-split` is used.
