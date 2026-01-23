"""Coulomb matrix feature generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from aimvee.datasets.geometry import read_xyz
from aimvee.utils import ensure_dir

try:
    import qml
except ImportError:  # pragma: no cover - runtime dependency
    qml = None


def _require_qml() -> None:
    if qml is None:
        raise ImportError(
            "qml is required to build Coulomb matrices. Install qml and retry."
        )


def coulomb_matrix_rep(
    numbers: np.ndarray, coords: np.ndarray, size: int
) -> np.ndarray:
    _require_qml()
    n_atoms = int(numbers.shape[0])
    if n_atoms > size:
        numbers = numbers[:size]
        coords = coords[:size]
        n_atoms = size
    mol = qml.Compound(xyz=None)
    mol.nuclear_charges = numbers
    mol.coordinates = coords
    mol.generate_coulomb_matrix(size=n_atoms, sorting="unsorted")
    rep = np.asarray(mol.representation, dtype=np.float32).ravel()
    expected = n_atoms * (n_atoms + 1) // 2
    if rep.shape[0] != expected:
        raise ValueError(
            f"Coulomb matrix dimension mismatch: expected {expected}, got {rep.shape[0]}"
        )
    if n_atoms == size:
        return rep
    full = np.zeros((size, size), dtype=np.float32)
    idx = 0
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            value = rep[idx]
            full[i, j] = value
            full[j, i] = value
            idx += 1
    tri = []
    for i in range(size):
        for j in range(i, size):
            tri.append(full[i, j])
    return np.asarray(tri, dtype=np.float32)


def build_cm_reps(rows: List[Tuple[Path, float]], size: int) -> np.ndarray:
    rep_len = size * (size + 1) // 2
    reps = np.zeros((len(rows), rep_len), dtype=np.float32)
    for idx, (xyz_path, _) in enumerate(rows):
        numbers, coords = read_xyz(xyz_path)
        reps[idx] = coulomb_matrix_rep(
            numbers.numpy(), coords.numpy(), size=size
        )
    return reps


def generate_cm(npz_path: str, output_dir: str) -> Path:
    """Generate Coulomb matrices for all samples in a .npz file."""
    _require_qml()
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    base_name = npz_path.stem
    print(f"\nProcessing {npz_path.name} ...")

    data = np.load(npz_path, allow_pickle=True)
    charges = data["Z"]
    coords_all = data["R"]

    n_samples = coords_all.shape[0]
    reps = []

    for i in range(n_samples):
        mol = qml.Compound(xyz=None)
        mol.nuclear_charges = charges
        mol.coordinates = coords_all[i]
        mol.generate_coulomb_matrix(size=len(charges), sorting="unsorted")
        reps.append(np.asarray(mol.representation).ravel())

    reps = np.asarray(reps)
    output_path = output_dir / f"{base_name}_CM.npy"
    np.save(output_path, reps)
    print(f"Saved CM to {output_path}")
    return output_path
