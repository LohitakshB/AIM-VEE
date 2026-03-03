# Experiments

Experiment runners live in `src/aimvee/experiments` and back the CLI commands.

## data_prep

- Command: `prepare-splits`
- Purpose: create train/val/test CSVs from metadata; can auto-build metadata from `E_exc_SS`.
- Outputs: `<splits-output>/<split>/train.csv`, `val.csv`, `test.csv`.

## qemfi

- Commands: `generate-cm`, `prep-qemfi`, `train-qemfi`
- Purpose: prepare and train the QeMFi surrogate.
- Outputs:
  - `models/qemfi_surrogate/best_model.pt`
  - `models/qemfi_surrogate/scaler_X.pkl`, `scaler_y.pkl`, optional `pca_X.pkl`

## schnet

- Command: `train-schnet`
- Purpose: SchNet baseline training on geometry splits.
- Outputs per split:
  - `models/schnet/<split>/best_model.pt`
  - `models/schnet/<split>/test_predictions.csv`

## rf_morgan

- Command: `train-rf-morgan`
- Purpose: random forest baseline on Morgan fingerprints.
- Outputs per split:
  - `models/rf_morgan/<split>/rf_morgan.pkl`
  - `models/rf_morgan/<split>/test_predictions.csv`

## chemprop

- Command: `train-chemprop`
- Purpose: train Chemprop baseline via external Chemprop CLI.
- Outputs per split: Chemprop checkpoint directory and saved predictions.

## mff_mlp

- Command: `train-mff-mlp`
- Purpose: train fusion MLP with QeMFi features + SchNet latent.
- Requires: trained QeMFi surrogate directory.
- Outputs per split:
  - `models/mff_mlp/<split>/best_model.pt`
  - `models/mff_mlp/<split>/best_schnet.pt`
  - `models/mff_mlp/<split>/qemfi_scaler.pkl`
  - `models/mff_mlp/<split>/test_predictions.csv`

## umff_mlp

- Command: `train-umff-mlp`
- Purpose: uncertainty-aware MFF with deep ensemble + isotonic calibration.
- Requires: trained QeMFi surrogate directory.
- Outputs per split:
  - `models/umff_mlp/<split>/ensemble_*/best_model.pt`
  - `models/umff_mlp/<split>/ensemble_*/best_schnet.pt`
  - `models/umff_mlp/<split>/qemfi_scaler.pkl`
  - `models/umff_mlp/<split>/isotonic_reg.pkl`
  - `models/umff_mlp/<split>/test_predictions.csv`

## infer_model

- Command: `infer-model`
- Purpose: prediction-only inference for `schnet`, `rf_morgan`, `chemprop`, `mff_mlp`, `umff_mlp`.
- Output:
  - `<output-dir>/predictions.csv` (includes `pred_mean`; UMFF adds uncertainty columns)

## evaluate_model

- Command: `evaluate-model`
- Purpose: per-model inference + metrics + plots.
- Outputs:
  - `<output-dir>/metrics_summary.json`
  - `<output-dir>/plots/*.png`

## evaluate

- Command: `evaluate`
- Purpose: aggregate statistics across many prediction files.
- Outputs:
  - JSON summary and optional CSV artifacts depending on flags.
