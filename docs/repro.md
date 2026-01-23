# Reproducibility checklist

This checklist mirrors the core pipeline used in the AIM-VEE experiments.

1) Generate Coulomb matrices for QeMFi

```
python -m aimvee.cli generate-cm --input-dir data/QeMFI_all/QeMFi --output-dir data/QeMFI_all/QeMFi_cm
```

2) Prepare QeMFi training data

```
python -m aimvee.cli prep-qemfi --qemfi-dir data/QeMFI_all/QeMFi --reps-dir data/QeMFI_all/QeMFi_cm --data-dir data/QeMFI_all/model_training_test_data
```

3) Train QeMFi surrogate

```
python -m aimvee.cli train-qemfi --data-dir data/QeMFI_all/model_training_test_data --output-dir models/qemfi_surrogate
```

4) Prepare QM9-GWBSE dataset splits

```
python -m aimvee.cli prepare-splits --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --output-dir data/QM9GWBSE/splits
```

5) Train MFF-MLP or baselines

```
python -m aimvee.cli train-mff-mlp --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/mff_mlp --qemfi-model-dir models/qemfi_surrogate
python -m aimvee.cli train-schnet --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/schnet
python -m aimvee.cli train-chemprop --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/chemprop
python -m aimvee.cli train-rf-morgan --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/rf_morgan
```

6) Evaluate and plot results

```
python -m aimvee.cli evaluate-qm9 --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --models-root models
python -m aimvee.cli plot-qm9 --results-dir outputs/qm9_testing --split-method random
```

Notes:
- `prepare-splits` can auto-generate `metadata_e_exc_ss.csv` if you supply `--exc-ss-dir`.
- Use `python -m aimvee.cli <command> --help` for full argument lists.
