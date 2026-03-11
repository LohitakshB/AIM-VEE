"""Shared utilities for AIM-VEE scripts."""

from __future__ import annotations

from pathlib import Path

import torch


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def select_device(device_arg: str) -> torch.device:
    """Select a torch device based on availability and user input."""
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
