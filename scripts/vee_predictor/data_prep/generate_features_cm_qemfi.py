import os
import numpy as np
from tqdm import tqdm
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aimvee.vee_predictor_utils.generate_cm import generate_cm


# Input/output directories
INPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/vee_predictor/QeMFi"
OUTPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/vee_predictor/QeMFi_CM"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Process all .npz files in INPUT_DIR."""
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]

    if not files:
        print("No .npz files found in the QeMFi directory.")
        return

    print(f"Found {len(files)} .npz files.")
    
    for file in files:
        npz_path = os.path.join(INPUT_DIR, file)
        generate_cm(npz_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()
