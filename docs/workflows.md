# Workflows

This page summarizes end-to-end workflows in AIM-VEE.

## Workflow A: QeMFi surrogate

1. Generate Coulomb matrices:

```bash
python -m aimvee.cli generate-cm --input-dir data/QeMFI_all/QeMFi --output-dir data/QeMFI_all/QeMFi_cm
```

2. Prepare surrogate arrays:

```bash
python -m aimvee.cli prep-qemfi --qemfi-dir data/QeMFI_all/QeMFi --reps-dir data/QeMFI_all/QeMFi_cm --data-dir data/QeMFI_all/model_training_test_data
```

3. Train surrogate:

```bash
python -m aimvee.cli train-qemfi --data-dir data/QeMFI_all/model_training_test_data --output-dir models/qemfi_surrogate
```

## Workflow B: QM9-GWBSE split generation

```bash
python -m aimvee.cli prepare-splits --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits
```

Notes:
- Use `--single-split --split-method random` (or another method) for one split strategy.
- Use `--split-method predefined --predefined-train ... --predefined-val ...` for externally defined splits.

## Workflow C: Model training

Train fusion models:

```bash
python -m aimvee.cli train-mff-mlp --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/mff_mlp --qemfi-model-dir models/qemfi_surrogate
python -m aimvee.cli train-umff-mlp --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/umff_mlp --qemfi-model-dir models/qemfi_surrogate
```

Train baselines:

```bash
python -m aimvee.cli train-schnet --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/schnet
python -m aimvee.cli train-chemprop --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/chemprop
python -m aimvee.cli train-rf-morgan --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/rf_morgan
```

## Workflow D: Inference/evaluation

Predictions only:

```bash
python -m aimvee.cli infer-model --model-type mff_mlp --model-dir models/mff_mlp/random --qemfi-model-dir models/qemfi_surrogate --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/mff_mlp_infer/random
```

Per-model evaluation (plots + metrics):

```bash
python -m aimvee.cli evaluate-model --model-type mff_mlp --model-dir models/mff_mlp/random --qemfi-model-dir models/qemfi_surrogate --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/mff_mlp_eval/random
```

Aggregate evaluation from prediction files:

```bash
python -m aimvee.cli evaluate --results-root outputs --output-json outputs/eval_summary.json
```
