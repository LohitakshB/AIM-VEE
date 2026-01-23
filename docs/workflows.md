# Workflows

This page summarizes the end-to-end workflows used in the project.

## QeMFi surrogate training

1) Build Coulomb matrices:

```
python -m aimvee.cli generate-cm --input-dir data/QeMFI_all/QeMFi --output-dir data/QeMFI_all/QeMFi_cm
```

2) Prepare training arrays:

```
python -m aimvee.cli prep-qemfi --qemfi-dir data/QeMFI_all/QeMFi --reps-dir data/QeMFI_all/QeMFi_cm --data-dir data/QeMFI_all/model_training_test_data
```

3) Train the surrogate:

```
python -m aimvee.cli train-qemfi --data-dir data/QeMFI_all/model_training_test_data --output-dir models/qemfi_surrogate
```

## QM9-GWBSE training

1) Prepare metadata (optional) and splits:

```
python -m aimvee.cli prepare-splits --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --output-dir data/QM9GWBSE/splits
```

2) Train models:

```
python -m aimvee.cli train-mff-mlp --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/mff_mlp --qemfi-model-dir models/qemfi_surrogate
python -m aimvee.cli train-schnet --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/schnet
python -m aimvee.cli train-chemprop --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/chemprop
python -m aimvee.cli train-rf-morgan --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/rf_morgan
```

## Evaluation and plotting

```
python -m aimvee.cli evaluate-qm9 --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --models-root models
python -m aimvee.cli plot-qm9 --results-dir outputs/qm9_testing --split-method random
```

Tips:
- Use `--models` on `evaluate-qm9` to restrict evaluation to a subset.
- The evaluation output includes per-model `predictions.csv` and `metrics.json`.
- If your QeMFi or MFF-MLP artifacts are stored elsewhere, pass
  `--qemfi-model-dir` and `--qemfi-scaler-path` to `evaluate-qm9`.
