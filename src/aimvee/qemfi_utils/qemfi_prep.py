"""Prepare QeMFi datasets for surrogate training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from aimvee.utils import ensure_dir


def _local_permutation(n_items: int, seed: int) -> np.ndarray:
    """Return a deterministic permutation for each molecule."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_items)


def expand_geom_fid_state_block(
    reps_block: np.ndarray,
    ev_block: np.ndarray,
    mol_ids_block: np.ndarray,
    geom_ids_block: np.ndarray,
    n_fids: int,
    n_states: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand geometry-level blocks into per-(geom, fid, state) rows."""
    n_geom, d_rep = reps_block.shape
    n_samples = n_geom * n_fids * n_states

    x_expanded = np.zeros((n_samples, d_rep + 2), dtype=float)
    y_expanded = np.zeros(n_samples, dtype=float)
    fid_ids = np.zeros(n_samples, dtype=np.int32)
    state_ids = np.zeros(n_samples, dtype=np.int32)
    mol_ids = np.zeros(n_samples, dtype=np.int64)
    geom_ids = np.zeros(n_samples, dtype=np.int64)

    k = 0
    for g in range(n_geom):
        for f in range(n_fids):
            for s in range(n_states):
                x_expanded[k, :d_rep] = reps_block[g]
                x_expanded[k, d_rep] = f
                x_expanded[k, d_rep + 1] = s

                y_expanded[k] = ev_block[g, f, s]

                fid_ids[k] = f
                state_ids[k] = s
                mol_ids[k] = mol_ids_block[g]
                geom_ids[k] = geom_ids_block[g]
                k += 1

    return x_expanded, y_expanded, fid_ids, state_ids, mol_ids, geom_ids


def _load_ev_block(qemfi_dir: Path, molecule: str) -> np.ndarray:
    npz_path = qemfi_dir / f"QeMFi_{molecule}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing QeMFi file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data["EV"]


def _load_reps(reps_dir: Path, molecule: str, method: str) -> np.ndarray:
    rep_path = reps_dir / f"QeMFi_{molecule}_{method}.npy"
    if not rep_path.exists():
        raise FileNotFoundError(f"Missing representation file: {rep_path}")
    return np.load(rep_path)


def prep_data(
    data_dir: str,
    molecules: Iterable[str],
    reps_dir: str,
    qemfi_dir: str,
    method: str,
    n_geom_per_mol: int = 15000,
    seed: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Prepare the {method}-based EV dataset with a shared geometry split."""
    data_root = Path(data_dir)
    reps_root = Path(reps_dir)
    qemfi_root = Path(qemfi_dir)
    ensure_dir(data_root)

    molecules = [m.strip() for m in molecules if m.strip()]
    if not molecules:
        raise ValueError("At least one molecule is required.")

    total_geoms = len(molecules) * n_geom_per_mol

    example_rep = _load_reps(reps_root, molecules[-1], method)
    max_dim = example_rep.shape[1]

    ev_sample = _load_ev_block(qemfi_root, molecules[0])
    n_fids = ev_sample.shape[1]
    n_states = ev_sample.shape[2]

    reps_all = np.zeros((total_geoms, max_dim), dtype=float)
    mol_ids_all = np.zeros(total_geoms, dtype=np.int64)
    geom_ids_all = np.zeros(total_geoms, dtype=np.int64)
    ev_all = np.zeros((total_geoms, n_fids, n_states), dtype=float)

    start_geom = 0

    for mol_id, mol in enumerate(molecules):
        end_geom = start_geom + n_geom_per_mol

        idx_local = _local_permutation(n_geom_per_mol, seed)

        reps_mol = _load_reps(reps_root, mol, method)
        d_m = reps_mol.shape[1]

        reps_mol_shuffled = reps_mol[idx_local, :]
        reps_all[start_geom:end_geom, :d_m] = reps_mol_shuffled

        ev_full = _load_ev_block(qemfi_root, mol)
        if ev_full.shape[1] != n_fids or ev_full.shape[2] != n_states:
            raise ValueError(
                f"EV shape mismatch for {mol}: expected (*,{n_fids},{n_states}), "
                f"got {ev_full.shape}"
            )

        ev_mol = ev_full[idx_local, :, :]
        ev_all[start_geom:end_geom, :, :] = ev_mol

        mol_ids_all[start_geom:end_geom] = mol_id
        geom_ids_all[start_geom:end_geom] = idx_local

        start_geom = end_geom

    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_geoms)

    reps_all = reps_all[perm]
    ev_all = ev_all[perm]
    mol_ids_all = mol_ids_all[perm]
    geom_ids_all = geom_ids_all[perm]

    n_train_geoms = 120_000
    n_val_geoms = 2_000
    n_test_geoms = total_geoms - n_train_geoms - n_val_geoms

    idx_train_rows = np.arange(0, n_train_geoms)
    idx_val_rows = np.arange(n_train_geoms, n_train_geoms + n_val_geoms)
    idx_test_rows = np.arange(n_train_geoms + n_val_geoms, total_geoms)

    reps_train_geom = reps_all[idx_train_rows]
    reps_val_geom = reps_all[idx_val_rows]
    reps_test_geom = reps_all[idx_test_rows]

    ev_train_geom = ev_all[idx_train_rows]
    ev_val_geom = ev_all[idx_val_rows]
    ev_test_geom = ev_all[idx_test_rows]

    mol_train_geom = mol_ids_all[idx_train_rows]
    mol_val_geom = mol_ids_all[idx_val_rows]
    mol_test_geom = mol_ids_all[idx_test_rows]

    geom_train_geom = geom_ids_all[idx_train_rows]
    geom_val_geom = geom_ids_all[idx_val_rows]
    geom_test_geom = geom_ids_all[idx_test_rows]

    (
        x_train,
        y_train,
        fid_train_ids,
        state_train_ids,
        _,
        _,
    ) = expand_geom_fid_state_block(
        reps_train_geom,
        ev_train_geom,
        mol_train_geom,
        geom_train_geom,
        n_fids,
        n_states,
    )
    (
        x_val,
        y_val,
        fid_val_ids,
        _,
        _,
        _,
    ) = expand_geom_fid_state_block(
        reps_val_geom,
        ev_val_geom,
        mol_val_geom,
        geom_val_geom,
        n_fids,
        n_states,
    )
    (
        x_test,
        y_test,
        fid_test_ids,
        _,
        _,
        _,
    ) = expand_geom_fid_state_block(
        reps_test_geom,
        ev_test_geom,
        mol_test_geom,
        geom_test_geom,
        n_fids,
        n_states,
    )

    high_fid_idx = n_fids - 1
    val_mask = fid_val_ids == high_fid_idx
    test_mask = fid_test_ids == high_fid_idx

    x_val = x_val[val_mask]
    y_val = y_val[val_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    np.save(data_root / f"{method}_train.npy", x_train)
    np.save(data_root / f"{method}_val.npy", x_val)
    np.save(data_root / f"{method}_test.npy", x_test)

    np.save(data_root / "EV_train.npy", y_train)
    np.save(data_root / "EV_val.npy", y_val)
    np.save(data_root / "EV_test.npy", y_test)

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        idx_train_rows,
        idx_val_rows,
        idx_test_rows,
        mol_ids_all,
        geom_ids_all,
        n_fids,
        n_states,
    )
