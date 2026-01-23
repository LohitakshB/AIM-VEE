# Python API

The public API lives in `src/aimvee/api.py` and is re-exported from `aimvee`.
These helpers mirror the CLI workflows.

## SplitConfig

Use `SplitConfig` to define dataset inputs and split settings for training.

```python
from aimvee import SplitConfig

split = SplitConfig(
    xyz_dir="data/QM9GWBSE/QM9_xyz_files",
    metadata="data/QM9GWBSE/metadata_e_exc_ss.csv",
    splits_output="data/QM9GWBSE/splits",
)
```

## Data prep

```python
from aimvee import build_metadata_from_exc_ss, prepare_splits

build_metadata_from_exc_ss(
    exc_ss_dir="data/QM9GWBSE/E_exc_SS",
    xyz_dir="data/QM9GWBSE/QM9_xyz_files",
    output_csv="data/QM9GWBSE/metadata_e_exc_ss.csv",
)

prepare_splits(split)
```

## QeMFi

```python
from aimvee import generate_qemfi_cm, prep_qemfi, train_qemfi

generate_qemfi_cm(
    input_dir="data/QeMFI_all/QeMFi",
    output_dir="data/QeMFI_all/QeMFi_cm",
)

prep_qemfi(
    qemfi_dir="data/QeMFI_all/QeMFi",
    reps_dir="data/QeMFI_all/QeMFi_cm",
    data_dir="data/QeMFI_all/model_training_test_data",
)

train_qemfi(
    data_dir="data/QeMFI_all/model_training_test_data",
    output_dir="models/qemfi_surrogate",
)
```

## Training

```python
from aimvee import train_mff_mlp, train_schnet, train_chemprop, train_rf_morgan

train_mff_mlp(split, output_dir="models/mff_mlp", qemfi_model_dir="models/qemfi_surrogate")
train_schnet(split, output_dir="models/schnet")
train_chemprop(split, output_dir="models/chemprop")
train_rf_morgan(split, output_dir="models/rf_morgan")
```

Notes:
- Evaluation and plotting are CLI-only (`evaluate-qm9`, `plot-qm9`).
- Most functions accept keyword arguments mirroring CLI flags.
