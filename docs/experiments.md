# Experiments

This page summarizes the experiment runners in `src/aimvee/experiments`.

## data_prep

- Builds train/val/test split CSVs from QM9 metadata.
- Can auto-generate metadata from `E_exc_SS` via `--exc-ss-dir`.
- Writes `data/QM9GWBSE/splits/<split>/train.csv` etc.

## qemfi

- `generate-cm`: converts QeMFi `.npz` files to Coulomb matrix representations.
- `prep-qemfi`: assembles train/val arrays for surrogate training.
- `train-qemfi`: trains the surrogate and writes scalers and checkpoints.

Artifacts:
- `models/qemfi_surrogate/best_model.pt`
- `models/qemfi_surrogate/scaler_X.pkl`, `scaler_y.pkl`, optional `pca_X.pkl`

## schnet

- Trains SchNet on QM9 splits using XYZ geometries.
- Writes `models/schnet/<split>/best_model.pt`.

## chemprop

- Calls the `chemprop` CLI to train a SMILES-based baseline.
- Requires `chemprop` to be installed and on PATH.

## rf_morgan

- Trains a random forest on Morgan fingerprints (RDKit).
- Writes `models/rf_morgan/<split>/rf_morgan.pkl`.

## mff_mlp

- Trains the fusion MFF-MLP using QeMFi surrogate outputs + SchNet latents.
- Requires pretrained QeMFi surrogate and SchNet checkpoint.
- Writes `models/mff_mlp/<split>/best_model.pt` and `best_schnet.pt`.

## qm9_testing

- `evaluate-qm9`: evaluates selected models on a QM9 test split.
- `plot-qm9`: generates parity/residual/error plots from predictions.

Artifacts:
- `outputs/qm9_testing/<split>/<model>/predictions.csv`
- `outputs/qm9_testing/<split>/<model>/metrics.json`
