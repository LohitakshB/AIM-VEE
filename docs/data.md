# Data layout

This project expects local datasets/artifacts in a non-versioned workspace layout.

## QeMFi data

```text
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

Relevant commands:
- `generate-cm` -> writes to `QeMFI_all/QeMFi_cm`
- `prep-qemfi` -> writes to `QeMFI_all/model_training_test_data`

## QM9-GWBSE data

```text
data/
  QM9GWBSE/
    QM9_xyz_files/          # XYZ geometries
    E_exc_SS/               # optional .dat files for auto metadata generation
    metadata_e_exc_ss.csv   # metadata input
    splits/
      random/
        train.csv
        val.csv
        test.csv
      bemis-murcko/
      size/
      tail-split/
```

Metadata columns used by split/training code:
- geometry id column (default `id` in split prep; training defaults to `geometry` after parser remap)
- `xyz_path`
- target column (default `lowest_excited_state` in split prep; training defaults to `output` after remap)
- `smiles` (required for Chemprop/RF Morgan unless derivable)

## Model artifacts

```text
models/
  qemfi_surrogate/
    best_model.pt
    scaler_X.pkl
    scaler_y.pkl
    pca_X.pkl (optional)

  mff_mlp/<split>/
    best_model.pt
    best_schnet.pt
    qemfi_scaler.pkl
    test_predictions.csv

  umff_mlp/<split>/
    ensemble_*/best_model.pt
    ensemble_*/best_schnet.pt
    qemfi_scaler.pkl
    isotonic_reg.pkl
    test_predictions.csv

  schnet/<split>/best_model.pt
  rf_morgan/<split>/rf_morgan.pkl
  chemprop/<split>/...        # produced by Chemprop CLI
```

## Runtime outputs

```text
outputs/
  *_infer/.../predictions.csv
  *_eval/.../metrics_summary.json
  *_eval/.../plots/*.png
  eval_summary.json            # from `evaluate`
```
