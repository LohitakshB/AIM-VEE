# AIM-VEE

AIM-VEE is a research codebase for predicting vertical excitation energies (VEE) using
a multi-fidelity fusion pipeline (QeMFi surrogate + SchNet latent + MFF-MLP) alongside
baseline models (SchNet, Chemprop, RF Morgan).

## Install

```
pip install -e .
```

Optional dependencies:
- `qml` for Coulomb matrices.
- `rdkit` for Morgan fingerprints and SMILES/scaffold processing.
- `torch_geometric` for SchNet-based models.
- `chemprop` if you want the Chemprop baseline.

## Quickstart (CLI)

Generate QeMFi Coulomb matrices:

```
python -m aimvee.cli generate-cm --input-dir data/QeMFI_all/QeMFi --output-dir data/QeMFI_all/QeMFi_cm
```

Prepare QeMFi training arrays:

```
python -m aimvee.cli prep-qemfi --qemfi-dir data/QeMFI_all/QeMFi --reps-dir data/QeMFI_all/QeMFi_cm --data-dir data/QeMFI_all/model_training_test_data
```

Train the QeMFi surrogate:

```
python -m aimvee.cli train-qemfi --data-dir data/QeMFI_all/model_training_test_data --output-dir models/qemfi_surrogate
```

Prepare split CSVs for QM9-GWBSE:

```
python -m aimvee.cli prepare-splits --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --output-dir data/QM9GWBSE/splits
```

Train MFF-MLP:

```
python -m aimvee.cli train-mff-mlp --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/mff_mlp --qemfi-model-dir models/qemfi_surrogate
```

Evaluate and plot QM9-GWBSE results:

```
python -m aimvee.cli evaluate-qm9 --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits
python -m aimvee.cli plot-qm9 --results-dir outputs/qm9_testing --split-method random
```

## Documentation

- `docs/structure.md`: repo layout and where to find components.
- `docs/repro.md`: reproducibility checklist for the main pipelines.
- `docs/cli.md`: CLI commands and examples.
- `docs/data.md`: expected data layout and artifacts.
- `docs/api.md`: Python API entrypoints.
- `docs/workflows.md`: end-to-end workflows and evaluation.
- `docs/experiments.md`: experiment runners and outputs.

## Notes

- `scripts/generate_metadata.py` is a standalone helper for building the QM9 metadata CSV.
- After installing, you can also run `aimvee <command>` instead of `python -m aimvee.cli`.
