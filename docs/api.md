# Python API

Public API is implemented in `src/aimvee/api.py` and re-exported from `aimvee`.

## Exported symbols

- `SplitConfig`
- `build_metadata_from_exc_ss`
- `prepare_splits`
- `train_schnet`
- `train_chemprop`
- `train_rf_morgan`
- `generate_qemfi_cm`
- `prep_qemfi`
- `train_qemfi`
- `train_mff_mlp`
- `infer_model`

## SplitConfig

```python
from aimvee import SplitConfig

split = SplitConfig(
    xyz_dir="data/QM9GWBSE/QM9_xyz_files",
    metadata="data/QM9GWBSE/metadata_e_exc_ss.csv",
    splits_output="data/QM9GWBSE/splits",
)
```

You can override split behavior via fields like:
- `split_method`
- `train_all_splits`
- `predefined_train` / `predefined_val` / `predefined_test`

## Data prep helpers

```python
from aimvee import build_metadata_from_exc_ss, prepare_splits

build_metadata_from_exc_ss(
    exc_ss_dir="data/QM9GWBSE/E_exc_SS",
    xyz_dir="data/QM9GWBSE/QM9_xyz_files",
    output_csv="data/QM9GWBSE/metadata_e_exc_ss.csv",
)

prepare_splits(split)
```

## QeMFi helpers

```python
from aimvee import generate_qemfi_cm, prep_qemfi, train_qemfi

generate_qemfi_cm(input_dir="data/QeMFI_all/QeMFi", output_dir="data/QeMFI_all/QeMFi_cm")
prep_qemfi(qemfi_dir="data/QeMFI_all/QeMFi", reps_dir="data/QeMFI_all/QeMFi_cm", data_dir="data/QeMFI_all/model_training_test_data")
train_qemfi(data_dir="data/QeMFI_all/model_training_test_data", output_dir="models/qemfi_surrogate")
```

## Training helpers

```python
from aimvee import train_mff_mlp, train_schnet, train_chemprop, train_rf_morgan

train_mff_mlp(split, output_dir="models/mff_mlp", qemfi_model_dir="models/qemfi_surrogate")
train_schnet(split, output_dir="models/schnet")
train_chemprop(split, output_dir="models/chemprop")
train_rf_morgan(split, output_dir="models/rf_morgan")
```

## Inference helper

```python
from aimvee import infer_model

infer_model(
    model_type="mff_mlp",   # schnet | rf_morgan | chemprop | mff_mlp | umff_mlp
    model_dir="models/mff_mlp/random",
    qemfi_model_dir="models/qemfi_surrogate",  # required for mff/umff
    input_csv="data/QM9GWBSE/splits/random/test.csv",
    output_dir="outputs/mff_mlp_infer/random",
)
```

Notes:
- API function kwargs mirror CLI args closely.
- There is currently no top-level API wrapper for `train_umff_mlp`, `evaluate-model`, or `evaluate`; use CLI for those.
