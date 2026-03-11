"""Dataset wrapper for QeMFi surrogate training."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class QemfiDataset(TorchDataset):
    """Dataset wrapper for QeMFi arrays.

    X shape: (N, d_rep + 2)
        [:, :d_rep] = CM features
        [:, -2]     = fidelity index
        [:, -1]     = state index
    y shape: (N,)
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.X = np.asarray(features, dtype=np.float32)
        self.y = np.asarray(targets, dtype=np.float32)

        self.d_rep = self.X.shape[1] - 2
        self.feats = self.X[:, :self.d_rep]
        self.fid_ids = self.X[:, -2].astype(np.int64)
        self.state_ids = self.X[:, -1].astype(np.int64)

        self.n_fids = int(np.max(self.fid_ids)) + 1
        self.n_states = int(np.max(self.state_ids)) + 1

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "feats": torch.tensor(self.feats[idx], dtype=torch.float32),
            "fid_id": torch.tensor(self.fid_ids[idx], dtype=torch.int64),
            "state_id": torch.tensor(self.state_ids[idx], dtype=torch.int64),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
        }
