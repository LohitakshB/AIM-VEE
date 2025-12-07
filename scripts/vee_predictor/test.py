# test_vee_predictor.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib  # to load saved scaler / PCA

from src.aimvee.models.VEE_predictor import vee_predictor as Model
from src.aimvee.vee_predictor_utils.load_dataset import Dataset
from src.aimvee.vee_predictor_utils.train_utils import eval_epoch


DATA_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/vee_predictor/Data"
MODEL_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/models/predictor"
MODEL_PATH = os.path.join(MODEL_DIR, "best_cm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_X.pkl")
PCA_PATH    = os.path.join(MODEL_DIR, "pca_X.pkl")

CM_TO_EV = 1.239841984e-4  # convert cm^-1 -> eV


def main():
    # 1) Load test CM features + EV targets
    X_test = np.load(os.path.join(DATA_DIR, "CM_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "EV_test.npy"))

    print("Test shapes:", X_test.shape, y_test.shape)

    # 2) Drop invalid targets (y < 0), same as train/val
    mask_test = y_test >= 0
    print(f"Dropping {np.sum(~mask_test)} invalid test rows (y < 0)")

    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    # 3) Separate representation (CM) from indices (fid, state)
    d_rep = X_test.shape[1] - 2
    X_test_rep = X_test[:, :d_rep]   # CM rep
    X_test_idx = X_test[:, d_rep:]   # (fid, state)

    # 4) Load scaler + PCA fitted on training data
    #    (Make sure your training script saved these!)
    scaler_X: StandardScaler = joblib.load(SCALER_PATH)
    pca: PCA = joblib.load(PCA_PATH)

    # 5) Apply scaling + PCA to test representation
    X_test_rep = scaler_X.transform(X_test_rep)
    X_test_rep = pca.transform(X_test_rep)

    new_d_rep = X_test_rep.shape[1]
    print("Test representation dim after PCA:", new_d_rep)

    # 6) Rejoin PCA rep with fid/state
    X_test = np.concatenate([X_test_rep, X_test_idx], axis=1)

    # 7) Build Dataset + DataLoader
    test_ds = Dataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    d_rep    = test_ds.d_rep
    n_fids   = test_ds.n_fids
    n_states = test_ds.n_states

    print(f"[TEST] d_rep={d_rep}, n_fids={n_fids}, n_states={n_states}")

    # 8) Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 9) Rebuild model with same hyperparameters as training
    model = Model(
        d_rep=d_rep,
        n_fids=n_fids,
        n_states=n_states,
        hidden_dim=512,
        emb_dim=32,
        dropout=0.2,
    ).to(device)

    # 10) Load best saved weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find model checkpoint at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model weights from: {MODEL_PATH}")

    # 11) Evaluate on test set
    test_mae_cm = eval_epoch(model, test_loader, device=device)
    test_mae_ev = test_mae_cm * CM_TO_EV

    print(
        f"\n=== FINAL TEST RESULTS ===\n"
        f"Test MAE: {test_mae_cm:.2f} cm^-1 ({test_mae_ev:.6f} eV)\n"
    )


if __name__ == "__main__":
    main()
