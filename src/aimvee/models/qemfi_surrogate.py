"""Neural surrogate model for QeMFi energy prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual MLP block: y = x + Dropout(GELU(BN(Linear(x))))."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QemfiSurrogate(nn.Module):
    """Feed-forward regression model with fidelity/state embeddings."""

    def __init__(
        self,
        d_rep: int,
        n_fids: int,
        n_states: int,
        hidden_dim: int = 512,
        emb_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.fid_emb = nn.Embedding(n_fids, emb_dim)
        self.state_emb = nn.Embedding(n_states, emb_dim)

        in_dim = d_rep + 2 * emb_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.backbone = nn.Sequential(
            ResBlock(hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, dropout=dropout),
        )

        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        feats: torch.Tensor,
        fid_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> torch.Tensor:
        fe = self.fid_emb(fid_idx)
        se = self.state_emb(state_idx)

        x = torch.cat([feats, fe, se], dim=-1)

        x = self.input_proj(x)
        x = self.backbone(x)

        out = self.out(x)

        return out.squeeze(-1)


# Backwards-compatible alias.
qemfi_surrogate = QemfiSurrogate
