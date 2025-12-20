import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aimvee.vee_predictor_utils.generate_cm import generate_cm
from src.aimvee.delta_learner_utils.xyz_utils import read_xyz


# ====== PATHS (EDIT THESE) ======
QUEST_CSV = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_parsed_CC3_aug-cc-pVTZ.csv"
# This should be the folder that contains the "structures/QUEST1/..." folders
GEOM_ROOT = os.path.dirname(QUEST_CSV)

OUTPUT_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/processed_quest"
os.makedirs(OUTPUT_DIR, exist_ok=True)



def main():
    # 1) Load QUEST CSV
    df = pd.read_csv(QUEST_CSV)

    # Optional: re-enforce filters (if not already done in previous step)
    # Keep only CC3/aug-cc-pVTZ
    mask_method = df["method"].str.contains("CC3") & df["method"].str.contains("aug-cc-pVTZ")
    df = df[mask_method].copy()

    # Drop unsafe
    mask_safe = ~(df["unsafe"].astype(str).str.lower() == "true")
    df = df[mask_safe].copy()

    # Drop missing geom_file
    mask_geom = df["geom_file"].notna() & (df["geom_file"].astype(str).str.strip() != "")
    df = df[mask_geom].copy()

    if df.empty:
        print("No valid rows after filtering — check your CSV/filters.")
        return

    print(f"Using {len(df)} QUEST rows after filtering.")

    # 2) First pass: figure out max number of atoms across all geometries
    max_atoms = 0
    geom_paths = []

    for _, row in df.iterrows():
        rel_path = row["geom_file"]
        full_path = os.path.join(GEOM_ROOT, rel_path)
        geom_paths.append(full_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Geometry file not found: {full_path}")

        Z, R = read_xyz(full_path)
        max_atoms = max(max_atoms, len(Z))

    print(f"Max atoms found in QUEST geometries: {max_atoms}")

    # 3) Second pass: build Coulomb matrices and collect targets
    cm_list = []
    y_ev_list = []
    mol_list = []
    state_list = []

    # Model will predict for fid_id = 4 (TVZP basis)
    fid_id = 4

    for (idx, row), geom_path in zip(df.iterrows(), geom_paths):
        Z, R = read_xyz(geom_path)
        cm_flat = generate_cm(Z, R, max_atoms)

        # Define a "state index" — here I use final_number, but you can change this
        state_id = int(row["final_number"])

        # If you want the same format as your QeMFi CM_* arrays (rep + fid + state),
        # append fid_id and state_id as extra columns:
        feats = np.concatenate([cm_flat, [fid_id, state_id]])

        cm_list.append(feats)
        y_ev_list.append(row["energy_eV"])
        mol_list.append(row["molecule"])
        state_list.append(state_id)

    X = np.vstack(cm_list)          # shape: (N, d_rep + 2)
    y = np.array(y_ev_list, float)  # shape: (N,)

    print(f"Final CM array shape: {X.shape}")
    print(f"Final energy array shape: {y.shape}")

    # 4) Save in whatever format you prefer
    # (a) Separate .npy files, similar to CM_train.npy / EV_train.npy style
    np.save(os.path.join(OUTPUT_DIR, "CM_QUEST.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "EV_QUEST.npy"), y)

    # (b) Combined .npz for convenience
    np.savez(
        os.path.join(OUTPUT_DIR, "QUEST_CM_EV.npz"),
        X=X,
        y=y,
        molecule=np.array(mol_list, dtype=object),
        state=np.array(state_list, dtype=int),
        fid_id=fid_id,
        max_atoms=max_atoms,
    )

    print(f"Saved:\n  CM_QUEST.npy\n  EV_QUEST.npy\n  QUEST_CM_EV.npz\nin {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
