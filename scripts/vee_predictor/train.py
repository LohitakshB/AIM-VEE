import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aimvee.vee_predictor_utils.generate_cm import generate_cm
from src.aimvee.models.VEE_predictor import vee_predictor as Model
from src.aimvee.vee_predictor_utils.load_dataset import Dataset
from src.aimvee.vee_predictor_utils.train_utils import train_epoch, eval_epoch
import joblib  # to save scaler / PCA

DATA_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/vee_predictor/Data_cm"


def main():
    #1 Load preprocessed CM features + EV targets
    X_train = np.load(os.path.join(DATA_DIR, "CM_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "EV_train.npy"))

    X_val   = np.load(os.path.join(DATA_DIR, "CM_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "EV_val.npy"))

    print("Loaded shapes:", X_train.shape, y_train.shape)

    #2 Drop invalid targets (y < 0)
    mask_train = y_train >= 0
    mask_val   = y_val >= 0

    print(f"Dropping {np.sum(~mask_train)} invalid train rows (y < 0)")
    print(f"Dropping {np.sum(~mask_val)} invalid val rows (y < 0)")

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    X_val = X_val[mask_val]
    y_val = y_val[mask_val]



    #3 Separate representation (CM) from indices (fid, state)
    d_rep = X_train.shape[1] - 2   # last 2 columns = fid, state

    X_train_rep = X_train[:, :d_rep]   # CM representation
    X_train_idx = X_train[:, d_rep:]   # fidelity + state

    X_val_rep = X_val[:, :d_rep]
    X_val_idx = X_val[:, d_rep:]



    #4 Scale CM representation using StandardScaler

    scaler_X = StandardScaler()
    X_train_rep = scaler_X.fit_transform(X_train_rep)
    X_val_rep   = scaler_X.transform(X_val_rep)



    #5 PCA reduction on representation

    N_COMPONENTS = 100

    print(f"Applying PCA reduction to {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS)
    X_train_rep = pca.fit_transform(X_train_rep)
    X_val_rep   = pca.transform(X_val_rep)

    new_d_rep = X_train_rep.shape[1]
    print("New representation dim:", new_d_rep)


    #6 Rejoin PCA representation with fid/state
    X_train = np.concatenate([X_train_rep, X_train_idx], axis=1)
    X_val   = np.concatenate([X_val_rep,   X_val_idx],   axis=1)



    #7 Build Dataset + DataLoaders
    train_ds = Dataset(X_train, y_train)
    val_ds   = Dataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    d_rep    = train_ds.d_rep     # updated rep dimension
    n_fids   = train_ds.n_fids
    n_states = train_ds.n_states

    print(f"d_rep={d_rep}, n_fids={n_fids}, n_states={n_states}")

    # Save scaler + PCA for test time
    joblib.dump(scaler_X, os.path.join(DATA_DIR, "scaler_X.pkl"))
    joblib.dump(pca,       os.path.join(DATA_DIR, "pca_X.pkl"))

    #8 Device + Model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = Model(
        d_rep=d_rep,
        n_fids=n_fids,
        n_states=n_states,
        hidden_dim=512,
        emb_dim=32,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val = float("inf")
    CM_TO_EV = 1.239841984e-4

 
    #9 Training loop
    for epoch in range(50):
        print("\nStarting epoch:", epoch)

        train_mae = train_epoch(model, train_loader, optimizer, device=device)
        val_mae   = eval_epoch(model, val_loader, device=device)

        scheduler.step(val_mae)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE: {train_mae:.2f} cm^-1 ({train_mae*CM_TO_EV:.6f} eV) | "
            f"Val MAE: {val_mae:.2f} cm^-1 ({val_mae*CM_TO_EV:.6f} eV)"
        )

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_cm_model.pt"))

            print(
                f" New BEST model saved with Val MAE = "
                f"{val_mae:.2f} cm^-1 ({val_mae*CM_TO_EV:.6f} eV)"
            )


if __name__ == "__main__":
    main()