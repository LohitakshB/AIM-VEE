import torch
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, X, y):
        """Dataset wrapper.

            X shape: (N, d_rep + 2)
                [:, :d_rep] = CM features
                [:, -2]     = fid index (int)
                [:, -1]     = state index (int)
        y shape: (N,)"""

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

        # d_rep: feature dimension
        self.d_rep = X.shape[1] - 2

        # separate internal arrays
        self.feats = self.X[:, :self.d_rep]

        # fidelity and state must be integers for embeddings
        self.fid_ids   = self.X[:, -2].astype(np.int64)
        self.state_ids = self.X[:, -1].astype(np.int64)

        # count categories
        self.n_fids   = int(np.max(self.fid_ids)) + 1
        self.n_states = int(np.max(self.state_ids)) + 1


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        return {
            "feats": torch.tensor(self.feats[idx]),         # float32
            "fid_id": torch.tensor(self.fid_ids[idx]),      # int64
            "state_id": torch.tensor(self.state_ids[idx]),  # int64
            "target": torch.tensor(self.y[idx])             # float32
        }
