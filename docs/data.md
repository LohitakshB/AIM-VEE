# Data layout

This project expects local data and artifacts in the following layout (not tracked by git).

## QeMFi data

```
data/
  QeMFI_all/
    QeMFi/                  # raw QeMFi .npz files
    QeMFi_cm/               # generated Coulomb matrices
    model_training_test_data/
      CM_train.npy
      CM_val.npy
      EV_train.npy
      EV_val.npy
      ...
```

Key commands:
- `generate-cm` writes Coulomb matrices into `QeMFi_cm/`.
- `prep-qemfi` writes training arrays into `model_training_test_data/`.

## QM9-GWBSE data

```
data/
  QM9GWBSE/
    QM9_xyz_files/          # XYZ geometries
    E_exc_SS/               # per-geometry .dat files (optional for metadata build)
    metadata_e_exc_ss.csv   # metadata CSV (id, xyz_path, target)
    splits/
      random/
        train.csv
        val.csv
        test.csv
      ...
```

Key commands:
- `prepare-splits` reads `metadata_e_exc_ss.csv` and writes `splits/`.
- If `metadata_e_exc_ss.csv` is missing, use `--exc-ss-dir` or
  `scripts/generate_metadata.py` to build it.

Expected metadata columns:
- `id` or `geometry_id`
- `xyz_path`
- `lowest_excited_state` (target)
- `smiles` (required for Chemprop/RF Morgan)

## Model artifacts and outputs

```
models/
  qemfi_surrogate/
  mff_mlp/
  schnet/
  chemprop/
  rf_morgan/

outputs/
```
