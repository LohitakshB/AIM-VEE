import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_mae = 0.0
    n_samples = 0

    for batch in loader:
        # move to device and set dtypes
        feats = batch["feats"].to(device).float()      # (B, d_rep)
        fid   = batch["fid_id"].to(device).long()      # (B,)
        state = batch["state_id"].to(device).long()    # (B,)
        y     = batch["target"].to(device).float()     # (B,)

        # forward
        pred = model(feats, fid, state)                # (B,)

  
        loss = F.l1_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        # MAE in cm^-1 for logging
        mae_batch = torch.mean(torch.abs(pred - y)).item()

        total_mae += mae_batch * batch_size
        n_samples += batch_size

    # return mean absolute error over dataset (cm^-1)
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
        
            # calculate MAE in cm^-1
            mae_batch = torch.mean(torch.abs(pred - y)).item()

            batch_size = y.size(0)
            total_mae += mae_batch * batch_size
            n_samples += batch_size

    # return mean MAE in cm^-1
    return total_mae / n_samples
