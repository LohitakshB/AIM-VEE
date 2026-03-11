"""Python API for AIM-VEE workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Union

from aimvee.data_utils.data_prep import iter_split_csvs
from aimvee.experiments import chemprop as chemprop_exp
from aimvee.experiments import data_prep as data_prep_exp
from aimvee.experiments import infer_model as infer_model_exp
from aimvee.experiments import mff_mlp as mff_mlp_exp
from aimvee.experiments import qemfi as qemfi_exp
from aimvee.experiments import rf_morgan as rf_morgan_exp
from aimvee.experiments import schnet as schnet_exp


def _to_path(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for dataset splits used across experiments."""

    xyz_dir: Path
    metadata: Path
    splits_output: Path
    id_column: str = "id"
    target_column: str = "lowest_excited_state"
    smiles_column: Optional[str] = "smiles"
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 0
    split_method: Optional[str] = None
    train_all_splits: bool = True
    predefined_train: Optional[Path] = None
    predefined_val: Optional[Path] = None
    predefined_test: Optional[Path] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz_dir", Path(self.xyz_dir))
        object.__setattr__(self, "metadata", Path(self.metadata))
        object.__setattr__(self, "splits_output", Path(self.splits_output))
        object.__setattr__(self, "predefined_train", _to_path(self.predefined_train))
        object.__setattr__(self, "predefined_val", _to_path(self.predefined_val))
        object.__setattr__(self, "predefined_test", _to_path(self.predefined_test))
        if self.smiles_column == "":
            object.__setattr__(self, "smiles_column", None)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "SplitConfig":
        """Build SplitConfig from a CLI argparse namespace."""
        return cls(
            xyz_dir=Path(args.xyz_dir),
            metadata=Path(args.metadata),
            splits_output=Path(args.splits_output),
            id_column=args.id_column,
            target_column=args.target_column,
            smiles_column=args.smiles_column if getattr(args, "smiles_column", None) else None,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            split_method=getattr(args, "split_method", None),
            train_all_splits=args.train_all_splits,
            predefined_train=_to_path(getattr(args, "predefined_train", None)),
            predefined_val=_to_path(getattr(args, "predefined_val", None)),
            predefined_test=_to_path(getattr(args, "predefined_test", None)),
        )


def _split_kwargs(split: SplitConfig) -> dict[str, object]:
    return {
        "xyz_dir": split.xyz_dir,
        "metadata": split.metadata,
        "splits_output": split.splits_output,
        "id_column": split.id_column,
        "target_column": split.target_column,
        "smiles_column": split.smiles_column,
        "train_frac": split.train_frac,
        "val_frac": split.val_frac,
        "seed": split.seed,
        "split_method": split.split_method,
        "train_all_splits": split.train_all_splits,
        "predefined_train": split.predefined_train,
        "predefined_val": split.predefined_val,
        "predefined_test": split.predefined_test,
    }


def _override_split(split: SplitConfig, **overrides: object) -> SplitConfig:
    return replace(split, **overrides)


def build_metadata_from_exc_ss(
    exc_ss_dir: Union[str, Path],
    xyz_dir: Union[str, Path],
    output_csv: Union[str, Path],
) -> None:
    """Create a metadata CSV by reading the first target per .dat file."""
    data_prep_exp.build_metadata_from_exc_ss(
        Path(exc_ss_dir), Path(xyz_dir), Path(output_csv)
    )


def prepare_splits(
    split: SplitConfig,
    *,
    exc_ss_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Generate split CSVs and optionally backfill metadata from E_exc_SS."""
    metadata_path = split.metadata
    if exc_ss_dir and not metadata_path.exists():
        build_metadata_from_exc_ss(exc_ss_dir, split.xyz_dir, metadata_path)

    list(
        iter_split_csvs(
            xyz_dir=split.xyz_dir,
            metadata_path=metadata_path,
            output_root=split.splits_output,
            id_column=split.id_column,
            target_column=split.target_column,
            smiles_column=split.smiles_column,
            train_frac=split.train_frac,
            val_frac=split.val_frac,
            seed=split.seed,
            split_method=split.split_method,
            train_all_splits=split.train_all_splits,
            predefined_train_csv=split.predefined_train,
            predefined_val_csv=split.predefined_val,
            predefined_test_csv=split.predefined_test,
        )
    )


def train_schnet(
    split: SplitConfig,
    *,
    output_dir: Union[str, Path],
    hidden_dim: int = 128,
    num_filters: int = 128,
    num_interactions: int = 6,
    num_gaussians: int = 50,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    readout: str = "add",
    dipole: bool = False,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "",
    id_column: str = "geometry",
    target_column: str = "output",
    smiles_column: Optional[str] = "smiles",
) -> None:
    split = _override_split(
        split,
        id_column=id_column,
        target_column=target_column,
        smiles_column=smiles_column,
    )
    args = argparse.Namespace(
        **_split_kwargs(split),
        output_dir=Path(output_dir),
        hidden_dim=hidden_dim,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        readout=readout,
        dipole=dipole,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    schnet_exp.run_schnet(args)


def train_chemprop(
    split: SplitConfig,
    *,
    output_dir: Union[str, Path],
    epochs: int = 50,
    batch_size: int = 50,
    hidden_size: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
    loss_function: str = "mse",
    metric: str = "mae",
    chemprop_args: Optional[str] = None,
    id_column: str = "geometry",
    target_column: str = "output",
    smiles_column: Optional[str] = "smiles",
) -> None:
    split = _override_split(
        split,
        id_column=id_column,
        target_column=target_column,
        smiles_column=smiles_column,
    )
    args = argparse.Namespace(
        **_split_kwargs(split),
        output_dir=Path(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        depth=depth,
        dropout=dropout,
        loss_function=loss_function,
        metric=metric,
        chemprop_args=chemprop_args or "",
    )
    chemprop_exp.run_chemprop(args)


def train_rf_morgan(
    split: SplitConfig,
    *,
    output_dir: Union[str, Path],
    radius: int = 2,
    n_bits: int = 2048,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "1.0",
    n_jobs: int = -1,
    model_seed: int = 13,
    id_column: str = "geometry",
    target_column: str = "output",
    smiles_column: Optional[str] = "smiles",
) -> None:
    split = _override_split(
        split,
        id_column=id_column,
        target_column=target_column,
        smiles_column=smiles_column,
    )
    args = argparse.Namespace(
        **_split_kwargs(split),
        output_dir=Path(output_dir),
        radius=radius,
        n_bits=n_bits,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        model_seed=model_seed,
    )
    rf_morgan_exp.run_rf_morgan(args)


def generate_qemfi_cm(
    input_dir: Union[str, Path], output_dir: Union[str, Path]
) -> None:
    args = argparse.Namespace(input_dir=Path(input_dir), output_dir=Path(output_dir))
    qemfi_exp.run_generate_cm(args)


def prep_qemfi(
    *,
    qemfi_dir: Union[str, Path],
    reps_dir: Union[str, Path],
    data_dir: Union[str, Path],
    method: str = "CM",
    molecules: str = (
        "urea,acrolein,alanine,sma,nitrophenol,urocanic,dmabn,thymine,o-hbdi"
    ),
    n_geom_per_mol: int = 15000,
) -> None:
    args = argparse.Namespace(
        qemfi_dir=Path(qemfi_dir),
        reps_dir=Path(reps_dir),
        data_dir=Path(data_dir),
        method=method,
        molecules=molecules,
        n_geom_per_mol=n_geom_per_mol,
    )
    qemfi_exp.run_prep_qemfi(args)


def train_qemfi(
    *,
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = "CM",
    pca_components: int = 100,
    batch_size: int = 256,
    epochs: int = 100,
    hidden_dim: int = 512,
    emb_dim: int = 32,
    dropout: float = 0.05,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    device: str = "",
) -> None:
    args = argparse.Namespace(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        method=method,
        pca_components=pca_components,
        batch_size=batch_size,
        epochs=epochs,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )
    qemfi_exp.run_train_qemfi(args)


def train_mff_mlp(
    split: SplitConfig,
    *,
    output_dir: Union[str, Path],
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    mlp_hidden_dim: int = 256,
    mlp_layers: int = 3,
    mlp_dropout: float = 0.1,
    qemfi_model_dir: Union[str, Path] = "models/aim_vee_model/qemfi_surrogate",
    qemfi_fidelity: int = 4,
    qemfi_batch_size: int = 2048,
    n_lowest_states: int = 10,
    schnet_hidden_dim: int = 128,
    schnet_num_filters: int = 128,
    schnet_num_interactions: int = 6,
    schnet_num_gaussians: int = 50,
    schnet_cutoff: float = 10.0,
    schnet_max_num_neighbors: int = 32,
    schnet_readout: str = "add",
    schnet_dipole: bool = False,
    schnet_batch_size: int = 32,
    device: str = "",
    id_column: str = "geometry",
    target_column: str = "output",
    smiles_column: Optional[str] = "smiles",
) -> None:
    split = _override_split(
        split,
        id_column=id_column,
        target_column=target_column,
        smiles_column=smiles_column,
    )
    args = argparse.Namespace(
        **_split_kwargs(split),
        output_dir=Path(output_dir),
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout,
        qemfi_model_dir=Path(qemfi_model_dir),
        qemfi_fidelity=qemfi_fidelity,
        qemfi_batch_size=qemfi_batch_size,
        n_lowest_states=n_lowest_states,
        schnet_hidden_dim=schnet_hidden_dim,
        schnet_num_filters=schnet_num_filters,
        schnet_num_interactions=schnet_num_interactions,
        schnet_num_gaussians=schnet_num_gaussians,
        schnet_cutoff=schnet_cutoff,
        schnet_max_num_neighbors=schnet_max_num_neighbors,
        schnet_readout=schnet_readout,
        schnet_dipole=schnet_dipole,
        schnet_batch_size=schnet_batch_size,
        device=device,
    )
    mff_mlp_exp.run_mff_mlp(args)


def infer_model(
    *,
    model_type: str,
    model_dir: Union[str, Path],
    input_csv: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "",
    batch_size: int = 32,
    hidden_dim: int = 128,
    num_filters: int = 128,
    num_interactions: int = 6,
    num_gaussians: int = 50,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    readout: str = "add",
    dipole: bool = False,
    radius: int = 2,
    n_bits: int = 2048,
    chemprop_args: Optional[str] = None,
    chemprop_checkpoint_dir: Optional[Union[str, Path]] = None,
    chemprop_checkpoint_path: Optional[Union[str, Path]] = None,
    schnet_hidden_dim: int = 128,
    schnet_num_filters: int = 128,
    schnet_num_interactions: int = 6,
    schnet_num_gaussians: int = 50,
    schnet_cutoff: float = 10.0,
    schnet_max_num_neighbors: int = 32,
    schnet_readout: str = "add",
    schnet_dipole: bool = False,
    qemfi_model_dir: Optional[Union[str, Path]] = None,
    qemfi_fidelity: int = 4,
    qemfi_batch_size: int = 2048,
    n_lowest_states: int = 10,
    schnet_batch_size: int = 32,
    mlp_hidden_dim: int = 256,
    mlp_layers: int = 3,
    mlp_dropout: float = 0.1,
) -> None:
    args = argparse.Namespace(
        model_type=model_type,
        model_dir=Path(model_dir),
        input_csv=Path(input_csv),
        output_dir=Path(output_dir),
        device=device,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        readout=readout,
        dipole=dipole,
        radius=radius,
        n_bits=n_bits,
        chemprop_args=chemprop_args,
        chemprop_checkpoint_dir=(
            Path(chemprop_checkpoint_dir) if chemprop_checkpoint_dir else None
        ),
        chemprop_checkpoint_path=(
            Path(chemprop_checkpoint_path) if chemprop_checkpoint_path else None
        ),
        schnet_hidden_dim=schnet_hidden_dim,
        schnet_num_filters=schnet_num_filters,
        schnet_num_interactions=schnet_num_interactions,
        schnet_num_gaussians=schnet_num_gaussians,
        schnet_cutoff=schnet_cutoff,
        schnet_max_num_neighbors=schnet_max_num_neighbors,
        schnet_readout=schnet_readout,
        schnet_dipole=schnet_dipole,
        qemfi_model_dir=Path(qemfi_model_dir) if qemfi_model_dir else None,
        qemfi_fidelity=qemfi_fidelity,
        qemfi_batch_size=qemfi_batch_size,
        n_lowest_states=n_lowest_states,
        schnet_batch_size=schnet_batch_size,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout,
    )
    infer_model_exp.run_infer_model(args)
