import os
import numpy as np
from dscribe.descriptors import ACSF
from ase import Atoms
from tqdm import tqdm

def generate_acsf(npz_path, output_dir):
    """Generate ACSF for all samples in a .npz file and save output"""
    
    filename = os.path.basename(npz_path)
    base_name = filename.replace(".npz", "")
    
    data = np.load(npz_path, allow_pickle=True)
    Z = data["Z"]          # Atomic numbers
    R_all = data["R"]      # Shape (N_samples, N_atoms, 3)
    
    # 1. Identify unique elements in this molecule
    unique_elements = np.unique(Z)
    
    # 2. Setup ACSF Descriptor
    acsf = ACSF(
    species=unique_elements.tolist(),
    r_cut=6,
    g2_params=[[1.0, 0.5], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]],
    

    g4_params=[
        [1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 4.0, 1.0],  
        [1.0, 1.0, -1.0], [1.0, 2.0, -1.0], [1.0, 4.0, -1.0] 
    ],
    

    )

    n_samples = R_all.shape[0]
    reps = []

    print(f"\nProcessing {filename} ...")
    for i in tqdm(range(n_samples), desc=f"Generating ACSF ({base_name})"):
        # DScribe uses ASE Atoms objects
        mol = Atoms(numbers=Z, positions=R_all[i])
        
        # Returns shape (N_atoms, N_features)
        atom_features = acsf.create(mol)
        
        # To get a Molecule-level vector (like CM), we sum the features of all atoms
        mol_vector = np.sum(atom_features, axis=0) 
        reps.append(mol_vector)

    reps = np.asarray(reps, dtype=np.float32) # Use float32 to save space

    output_path = os.path.join(output_dir, f"{base_name}_ACSF.npy")
    np.save(output_path, reps)
    print(f"Saved ACSF to {output_path} | Shape: {reps.shape}")

