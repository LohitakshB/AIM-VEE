"""Generate Coulomb matrix representations for QeMFi samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import qml
from tqdm import tqdm

from aimvee.utils import ensure_dir


def generate_cm(npz_path: str, output_dir: str) -> None:
    """Generate Coulomb matrices for all samples in a .npz file."""
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

    for i in tqdm(range(n_samples), desc=f"Generating CM ({base_name})"):
        mol = qml.Compound(xyz=None)
        mol.nuclear_charges = charges
        mol.coordinates = coords_all[i]
        mol.generate_coulomb_matrix(size=len(charges), sorting="unsorted")
        reps.append(np.asarray(mol.representation).ravel())

    reps = np.asarray(reps)
    output_path = output_dir / f"{base_name}_CM.npy"
    np.save(output_path, reps)
    print(f"Saved CM to {output_path}")
