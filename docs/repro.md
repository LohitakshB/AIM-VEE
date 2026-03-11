# Reproducibility checklist

This checklist mirrors the intended end-to-end pipeline.

## 1) Generate Coulomb matrices for QeMFi

```bash
python -m aimvee.cli generate-cm --input-dir data/QeMFI_all/QeMFi --output-dir data/QeMFI_all/QeMFi_cm
```

## 2) Prepare QeMFi training data

```bash
python -m aimvee.cli prep-qemfi --qemfi-dir data/QeMFI_all/QeMFi --reps-dir data/QeMFI_all/QeMFi_cm --data-dir data/QeMFI_all/model_training_test_data
```

## 3) Train QeMFi surrogate

```bash
python -m aimvee.cli train-qemfi --data-dir data/QeMFI_all/model_training_test_data --output-dir models/qemfi_surrogate
```

## 4) Prepare QM9-GWBSE splits

```bash
python -m aimvee.cli prepare-splits --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits
```

## 5) Train target models

```bash
python -m aimvee.cli train-mff-mlp --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/mff_mlp --qemfi-model-dir models/qemfi_surrogate
python -m aimvee.cli train-umff-mlp --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/umff_mlp --qemfi-model-dir models/qemfi_surrogate
python -m aimvee.cli train-schnet --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/schnet
python -m aimvee.cli train-chemprop --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/chemprop
python -m aimvee.cli train-rf-morgan --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata.csv --splits-output data/QM9GWBSE/splits --output-dir models/rf_morgan
```

## 6) Run inference/evaluation

```bash
python -m aimvee.cli infer-model --model-type mff_mlp --model-dir models/mff_mlp/random --qemfi-model-dir models/qemfi_surrogate --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/mff_mlp_infer/random
python -m aimvee.cli evaluate-model --model-type mff_mlp --model-dir models/mff_mlp/random --qemfi-model-dir models/qemfi_surrogate --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/mff_mlp_eval/random
python -m aimvee.cli evaluate --results-root outputs --output-json outputs/eval_summary.json
```

## Notes

- `prepare-splits` can auto-create metadata if `--metadata` is missing and `--exc-ss-dir` is provided.
- Most train commands run all split strategies by default; add `--single-split` to constrain runs.
- Use `python -m aimvee.cli <command> --help` for exact argument defaults.
