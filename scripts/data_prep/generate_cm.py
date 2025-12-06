#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:13:27 2024

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
Script to generate unsorted CM representations for all 15,000 samples of a given molecule from the QeMFi database.
'''


import os
import numpy as np
import qml
from tqdm import tqdm

# Input and Output folders
INPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/QeMFi"
OUTPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/QeMFi_cm"

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_npz_file(npz_path):
    """Generate Coulomb Matrix for all samples in a .npz file and save output"""
    
    filename = os.path.basename(npz_path)               # ex: QeMFi_urea.npz
    base_name = filename.replace(".npz", "")            # ex: QeMFi_urea
    
    print(f"\nProcessing {filename} ...")

    data = np.load(npz_path, allow_pickle=True)
    Z = data["Z"]                       # atomic numbers
    R_all = data["R"]                   # coords: shape (N_samples, N_atoms, 3)

    n_samples = R_all.shape[0]
    reps = []

    for i in tqdm(range(n_samples), desc=f"Generating CM ({base_name})"):
        mol = qml.Compound(xyz=None)
        mol.nuclear_charges = Z
        mol.coordinates = R_all[i]
        mol.generate_coulomb_matrix(size=len(Z), sorting="unsorted")
        reps.append(mol.representation)

    reps = np.asarray(reps)

    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_CM.npy")
    np.save(output_path, reps)

    print(f" Saved CM to {output_path}")


def main():
    """Loop through ALL .npz files in INPUT_DIR"""
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]

    if not files:
        print("No .npz files found in the QeMFi directory.")
        return

    print(f"Found {len(files)} .npz files.")
    
    for file in files:
        npz_path = os.path.join(INPUT_DIR, file)
        process_npz_file(npz_path)


if __name__ == "__main__":
    main()
