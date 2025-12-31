import os
import numpy as np
import qml
from tqdm import tqdm

def generate_cm(npz_path,output_dir):   
    """Generate Coulomb Matrix for all samples in a .npz file and save output"""
    
    filename = os.path.basename(npz_path)               # e.g. QeMFi_urea.npz
    base_name = filename.replace(".npz", "")         # e.g. QeMFi_urea
    
    print(f"\nProcessing {filename} ...")

    data = np.load(npz_path, allow_pickle=True)
    Z = data["Z"]                       # atomic numbers
    R_all = data["R"]                   # coords: (N_samples, N_atoms, 3)

    n_samples = R_all.shape[0]
    reps = []

    for i in tqdm(range(n_samples), desc=f"Generating CM ({base_name})"):
        mol = qml.Compound(xyz=None)
        mol.nuclear_charges = Z
        mol.coordinates = R_all[i]
        mol.generate_coulomb_matrix(size=len(Z), sorting="unsorted")
        reps.append(mol.representation)

    reps = np.asarray(reps)

    output_path = os.path.join(output_dir, f"{base_name}_CM.npy")
    np.save(output_path, reps)

    print(f"Saved CM to {output_path}")