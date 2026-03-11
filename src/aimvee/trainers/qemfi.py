"""Training utilities for the QeMFi surrogate model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_scale: Optional[float] = None,
) -> float:
    """Run one training epoch and return MAE in cm^-1."""
    model.train()
    total_mae = 0.0
    n_samples = 0

    for batch in loader:
        feats = batch["feats"].to(device).float()
        fid = batch["fid_id"].to(device).long()
        state = batch["state_id"].to(device).long()
        target = batch["target"].to(device).float()

        pred = model(feats, fid, state)
        loss = F.l1_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        mae_batch = torch.mean(torch.abs(pred - target)).item()
        if target_scale is not None:
            mae_batch *= target_scale

        total_mae += mae_batch * batch_size
        n_samples += batch_size

    return total_mae / n_samples


def eval_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_scale: Optional[float] = None,
) -> float:
    """Evaluate one epoch and return MAE in cm^-1."""
    model.eval()
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            feats = batch["feats"].to(device).float()
            fid = batch["fid_id"].to(device).long()
            state = batch["state_id"].to(device).long()
            target = batch["target"].to(device).float()

            pred = model(feats, fid, state)

            mae_batch = torch.mean(torch.abs(pred - target)).item()
            if target_scale is not None:
                mae_batch *= target_scale

            batch_size = target.size(0)
            total_mae += mae_batch * batch_size
            n_samples += batch_size

    return total_mae / n_samples
