#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:16:28 2024

@author: vvinod
MIT License

Copyright (c) [2024] [Vivin Vinod]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
'''
Script to generate unsorted Global SLATM representations for all 15,000 samples of a given molecule from the QeMFi database.
'''

import os
import numpy as np
import qml
from qml.representations import get_slatm_mbtypes
from tqdm import tqdm

# qml hack for older versions
np.int = int  # needed for qml compatibility

# Input and output directories
INPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/QeMFi"
OUTPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/QeMFi_slatm"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_npz_file(npz_path):
    """
    Generate global SLATM representations for a single QeMFi .npz file
    and save them as a .npy file in OUTPUT_DIR.
    """
    filename = os.path.basename(npz_path)      # e.g. QeMFi_urea.npz
    base_name = filename.replace(".npz", "")   # e.g. QeMFi_urea

    print(f"Processing {filename} ...")

    data = np.load(npz_path, allow_pickle=True)
    Zs = data["Z"]          # atomic numbers, shape (n_atoms,)
    coords_all = data["R"]  # coords, shape (n_samples, n_atoms, 3)

    n_samples = coords_all.shape[0]

    # 1. Build qml.Compound objects
    compounds = []
    for i in tqdm(range(n_samples), desc=f"Loading compounds ({base_name})"):
        mol = qml.Compound(xyz=None)
        mol.nuclear_charges = Zs
        mol.coordinates = coords_all[i]
        compounds.append(mol)

    # 2. Get SLATM many-body types (mbtypes) across all molecules in this file
    mbtypes = get_slatm_mbtypes(
        np.array(
            [mol.nuclear_charges for mol in tqdm(compounds,
                                                 desc=f"Computing mbtypes ({base_name})")]
        )
    )

    # 3. Generate global SLATM representation for each geometry
    for mol in tqdm(compounds, desc=f"Generating SLATM ({base_name})"):
        mol.generate_slatm(mbtypes, local=False)

    # 4. Collect representations into a single array
    X_slatm = np.asarray(
        [mol.representation for mol in tqdm(compounds,
                                            desc=f"Collecting reps ({base_name})")]
    )

    # 5. Save to output folder
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_SLATM.npy")
    np.save(out_path, X_slatm)
    print(f"Saved SLATM to {out_path}")


def main():
    # Find all .npz files in the input directory
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]

    if not files:
        print("❌ No .npz files found in the QeMFi folder.")
        return

    print(f"Found {len(files)} .npz files in {INPUT_DIR}")

    for fname in files:
        npz_path = os.path.join(INPUT_DIR, fname)
        process_npz_file(npz_path)


if __name__ == "__main__":
    main()
