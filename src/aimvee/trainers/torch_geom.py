"""Training loops for torch-geometric models."""

from __future__ import annotations

import torch


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.z, batch.pos, batch.batch).view(-1)
        loss = torch.nn.functional.l1_loss(preds, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def eval_epoch(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.z, batch.pos, batch.batch).view(-1)
            loss = torch.nn.functional.l1_loss(preds, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)
