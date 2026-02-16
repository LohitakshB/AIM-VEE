# CLI reference

After installation, use either `aimvee <command>` or `python -m aimvee.cli <command>`.
Run `--help` on any command for full arguments.

## Core commands

- `generate-cm`: generate Coulomb matrices for QeMFi `.npz` files.
- `prep-qemfi`: build QeMFi training arrays from CM representations.
- `train-qemfi`: train the QeMFi surrogate model.
- `prepare-splits`: build QM9-GWBSE train/val/test CSV splits.
- `train-mff-mlp`: train the multi-fidelity fusion MLP.

## Baseline commands

- `train-schnet`: SchNet baseline.
- `train-chemprop`: Chemprop baseline (requires `chemprop` CLI).
- `train-rf-morgan`: random forest baseline on Morgan fingerprints.

## Examples

Generate splits and train SchNet:

```
python -m aimvee.cli prepare-splits --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --output-dir data/QM9GWBSE/splits
python -m aimvee.cli train-schnet --xyz-dir data/QM9GWBSE/QM9_xyz_files --metadata data/QM9GWBSE/metadata_e_exc_ss.csv --splits-output data/QM9GWBSE/splits --output-dir models/schnet
```
