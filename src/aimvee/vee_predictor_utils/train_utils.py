# train.py
import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, device, mean_y=None, std_y=None):
    model.train()
    total_mae = 0.0
    n_samples = 0

    for batch in loader:
        # Move to device + set correct dtypes
        feats = batch["feats"].to(device).float()      # (B, d_rep)
        fid   = batch["fid_id"].to(device).long()      # (B,)
        state = batch["state_id"].to(device).long()    # (B,)
        y     = batch["target"].to(device).float()     # (B,)

        # Forward
        pred = model(feats, fid, state)                # (B,)

        # --- Loss: use scaled y if mean_y/std_y provided ---
        if (mean_y is not None) and (std_y is not None):
            # scale both pred and y using the same stats
            pred_scaled = (pred - mean_y) / std_y
            y_scaled    = (y   - mean_y) / std_y
            loss = F.l1_loss(pred_scaled, y_scaled)
        else:
            # fallback: plain L1 in original units
            loss = F.l1_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        # MAE in REAL cm^-1 for logging
        mae_batch = torch.mean(torch.abs(pred - y)).item()

        total_mae += mae_batch * batch_size
        n_samples += batch_size

    # return mean absolute error over entire dataset (in cm^-1)
    return total_mae / n_samples


def eval_epoch(model, loader, device, mean_y=None, std_y=None):
    model.eval()
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            feats = batch["feats"].to(device).float()
            fid   = batch["fid_id"].to(device).long()
            state = batch["state_id"].to(device).long()
            y     = batch["target"].to(device).float()

            pred = model(feats, fid, state)

            # We still want the model trained with the same loss form as train_epoch,
            # so if you want to be super-consistent you could recompute the scaled loss here,
            # but for evaluation we care about REAL MAE:
            mae_batch = torch.mean(torch.abs(pred - y)).item()

            batch_size = y.size(0)
            total_mae += mae_batch * batch_size
            n_samples += batch_size

    # Return mean MAE in real cm^-1
    return total_mae / n_samples
