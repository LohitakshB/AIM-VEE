import os
import sys
import math
import numpy as np
import pandas as pd
import joblib
import qml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Inputs
QUEST_CSV = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_parsed_CC3_aug-cc-pVTZ.csv"
QUEST_ROOT = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_db"
MODEL_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/models/predictor"

# Output
OUT_NPY = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_CC3.npy"
OUT_ENERGY_NPY = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/delta_learner/QUEST_CC3_energy_eV.npy"

# Fidelity is fixed 
FID_ID = 4




def _cm_from_xyz(xyz_path: str, size: int) -> np.ndarray:
    mol = qml.Compound(xyz=xyz_path)
    mol.generate_coulomb_matrix(size=size, sorting="row-norm")
    rep = np.asarray(mol.representation, dtype=np.float32).ravel()
    return rep


def _pad_or_trim(vec: np.ndarray, target_len: int) -> np.ndarray:
    if vec.shape[0] < target_len:
        return np.pad(vec, (0, target_len - vec.shape[0]), mode="constant")
    if vec.shape[0] > target_len:
        return vec[:target_len]
    return vec


def _assign_state_ids(df: pd.DataFrame) -> pd.Series:
    """
    Assign state_id within each molecule:
    state_id=1 is the lowest energy (per molecule),
    then 2, 3, ... by increasing energy_eV.
    """
    df = df.copy()
    df["state_id"] = (
        df.groupby("molecule")["energy_eV"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return df["state_id"]


def main():
    df = pd.read_csv(QUEST_CSV)
    if "geom_file" not in df.columns or "energy_eV" not in df.columns:
        raise ValueError("QUEST_CSV must include geom_file and energy_eV columns.")

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_X.pkl"))

    raw_dim = int(scaler.n_features_in_)
    cm_size = 15

    df["state_id"] = _assign_state_ids(df)

    cm_rows = []
    fid_state_rows = []
    energy_rows = []

    for _, row in df.iterrows():
        geom_rel = str(row["geom_file"]).strip()
        if not geom_rel:
            continue
        xyz_path = os.path.join(QUEST_ROOT, geom_rel)
        if not os.path.exists(xyz_path):
            continue

        cm = _cm_from_xyz(xyz_path, size=cm_size)
        cm = _pad_or_trim(cm, raw_dim)

        cm_rows.append(cm)
        fid_state_rows.append([FID_ID, int(row["state_id"])])
        energy_rows.append(float(row["energy_eV"]))

    if not cm_rows:
        raise RuntimeError("No valid rows found (check geom_file paths).")

    X_raw = np.asarray(cm_rows, dtype=np.float32)
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)

    fid_state = np.asarray(fid_state_rows, dtype=np.int64)
    X_final = np.concatenate([X_pca, fid_state], axis=1)

    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
    np.save(OUT_NPY, X_final)
    np.save(OUT_ENERGY_NPY, np.asarray(energy_rows, dtype=np.float32))
    print("Saved:", OUT_NPY, "shape:", X_final.shape)


if __name__ == "__main__":
    main()
