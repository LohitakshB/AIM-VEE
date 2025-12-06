import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

from load_dataset import Dataset   
from model import Model
from train import train_epoch, eval_epoch


DATA_DIR = "/Users/lohitakshbadarala/Desktop/AIM-VEE/data/Data"


def main():
    # 1) Load preprocessed CM features + EV targets
    X_train = np.load(os.path.join(DATA_DIR, "CM_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "EV_train.npy"))

    X_val   = np.load(os.path.join(DATA_DIR, "CM_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "EV_val.npy"))


    # 2) Scale ONLY the representation part (first d_rep columns)
    #    Assume last 2 columns are: fid, state (do NOT scale those)

    d_rep = X_train.shape[1] - 2  # last 2 = fid, state

    scaler_X = StandardScaler()
    # fit on train feats
    X_train[:, :d_rep] = scaler_X.fit_transform(X_train[:, :d_rep])
    # apply same scaling to val feats
    X_val[:, :d_rep]   = scaler_X.transform(X_val[:, :d_rep])

    # 3) Build Datasets & DataLoaders
    #    Dataset should split into (feats, fid_idx, state_idx, target)
    #    and expose d_rep, n_fids, n_states attributes.
    train_ds = Dataset(X_train, y_train)
    val_ds   = Dataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    # Infer representation dimension & category counts from dataset
    d_rep    = train_ds.d_rep
    n_fids   = train_ds.n_fids
    n_states = train_ds.n_states

    # 4) Set up device & model
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
        hidden_dim=256,
        emb_dim=16
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    best_val = float("inf")

    # conversion: cm^-1 → eV
    CM_TO_EV = 1.239841984e-4

    # 5) Training loop
    for epoch in range(100):
        print("starting epoch:", epoch)

        # train_epoch / eval_epoch should return MAE in cm^-1
        train_mae = train_epoch(model, train_loader, optimizer, device=device)
        val_mae   = eval_epoch(model, val_loader, device=device)

        train_mae_ev = train_mae * CM_TO_EV
        val_mae_ev   = val_mae * CM_TO_EV

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE: {train_mae:.2f} cm^-1 ({train_mae_ev:.6f} eV) | "
            f"Val MAE: {val_mae:.2f} cm^-1 ({val_mae_ev:.6f} eV)"
        )

        # Save best model (based on cm^-1 MAE)
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_cm_model.pt"))
            print(
                f" New best model saved with Val MAE = "
                f"{val_mae:.2f} cm^-1 ({val_mae_ev:.6f} eV)"
            )


if __name__ == "__main__":
    main()
