# train.py
import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        # Move to device + set correct dtypes
        feats = batch["feats"].to(device).float()           # (B, d_rep)
        fid   = batch["fid_id"].to(device).long()           # (B,)   for nn.Embedding
        state = batch["state_id"].to(device).long()         # (B,)   for nn.Embedding
        y     = batch["target"].to(device).float()          # (B,)

        # Forward
        pred = model(feats, fid, state)                     # (B,)

        # L1 = MAE (in same units as y, e.g. cm^-1)
        loss = F.l1_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        n_samples  += batch_size

    # return mean absolute error over entire dataset
    return total_loss / n_samples


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            feats = batch["feats"].to(device).float()
            fid   = batch["fid_id"].to(device).long()
            state = batch["state_id"].to(device).long()
            y     = batch["target"].to(device).float()

            pred = model(feats, fid, state)

            loss = F.l1_loss(pred, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            n_samples  += batch_size

    return total_loss / n_samples
