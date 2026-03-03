# CLI reference

After installation, use either:
- `aimvee <command>`
- `python -m aimvee.cli <command>`

For details on any command:

```bash
python -m aimvee.cli <command> --help
```

## Commands

### QeMFi pipeline

- `generate-cm`
  - Generate Coulomb matrices from QeMFi `.npz` files.
  - Required: `--input-dir`, `--output-dir`

- `prep-qemfi`
  - Build train/val arrays from QeMFi raw data + CM reps.
  - Required: `--qemfi-dir`, `--reps-dir`, `--data-dir`
  - Common options: `--method`, `--molecules`, `--n-geom-per-mol`

- `train-qemfi`
  - Train the QeMFi surrogate.
  - Required: `--data-dir`, `--output-dir`
  - Common options: `--pca-components`, `--batch-size`, `--epochs`, `--hidden-dim`, `--emb-dim`, `--dropout`, `--lr`, `--weight-decay`, `--device`

### Split preparation

- `prepare-splits`
  - Generate train/val/test split CSVs from metadata.
  - Common options: `--xyz-dir`, `--metadata`, `--output-dir` (alias for internal splits output), `--split-method`, `--train-all-splits` / `--single-split`, `--predefined-train`, `--predefined-val`, `--predefined-test`
  - Can auto-build metadata from `--exc-ss-dir` when metadata is missing.

### Training commands

All model training commands below share split/dataset arguments from the split system (`--xyz-dir`, `--metadata`, `--splits-output`, split controls, id/target/smiles column options).

- `train-schnet`
  - Trains SchNet baseline.

- `train-rf-morgan`
  - Trains RF on Morgan fingerprints.

- `train-chemprop`
  - Trains Chemprop via external CLI.
  - Requires `chemprop` in PATH.

- `train-mff-mlp`
  - Trains MFF-MLP on QeMFi + SchNet features.
  - Requires `--qemfi-model-dir`.

- `train-umff-mlp`
  - Trains UMFF-MLP deep ensemble with isotonic uncertainty calibration.
  - Requires `--qemfi-model-dir`.
  - Common extra options: `--ensemble-size`, `--ensemble-seed`.

### Inference and evaluation

- `infer-model`
  - Runs inference and writes `<output-dir>/predictions.csv`.
  - `--model-type` choices: `schnet`, `rf_morgan`, `chemprop`, `mff_mlp`, `umff_mlp`
  - For `mff_mlp` / `umff_mlp`, pass `--qemfi-model-dir`.

- `evaluate-model`
  - Runs inference and writes metrics + plots.
  - `--model-type` choices: `qemfi`, `schnet`, `rf_morgan`, `chemprop`, `mff_mlp`, `umff_mlp`
  - For `qemfi`, pass `--qemfi-data-dir` (prepared arrays dir with `CM_test.npy` and `EV_test.npy`).
  - For `qemfi`, optional `--qemfi-eval-batch-size` controls prediction batching.
  - Outputs:
    - `<output-dir>/metrics_summary.json`
    - `<output-dir>/plots/*.png`

- `evaluate`
  - Aggregates saved prediction files across result directories.
  - Useful for experiment-level summaries and statistical comparisons.

## Copy-paste examples

Prepare splits and train SchNet:

```bash
python -m aimvee.cli prepare-splits --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits
python -m aimvee.cli train-schnet --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/schnet
```

Inference per model:

```bash
python -m aimvee.cli infer-model --model-type schnet --model-dir models/schnet/random --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/schnet_infer/random
python -m aimvee.cli infer-model --model-type rf_morgan --model-dir models/rf_morgan/random --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/rf_infer/random
python -m aimvee.cli infer-model --model-type chemprop --model-dir models/chemprop/random --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/chemprop_infer/random
python -m aimvee.cli infer-model --model-type mff_mlp --model-dir models/mff_mlp/random --qemfi-model-dir models/qemfi_surrogate --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/mff_mlp_infer/random
python -m aimvee.cli infer-model --model-type umff_mlp --model-dir models/umff_mlp/random --qemfi-model-dir models/qemfi_surrogate --input-csv data/QM9GWBSE/splits/random/test.csv --output-dir outputs/umff_infer/random
```
