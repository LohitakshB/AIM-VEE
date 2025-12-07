import torch
import torch.nn as nn
from .ResBlock import ResBlock

class vee_predictor(nn.Module):
    def __init__(
        self,
        d_rep: int,
        n_fids: int,
        n_states: int,
        hidden_dim: int = 512,
        emb_dim: int = 32,
        dropout: float = 0.2,
    ):
        """
        Feed-forward regression model with:
          - embeddings for fidelity and state
          - initial projection to hidden_dim
          - 3 residual MLP blocks
          - final linear head

        Residual blocks help deeper networks train more stably and
        usually reduce MAE for molecular regression tasks.
        """
        super().__init__()

        # Embeddings for fidelity and state indices
        self.fid_emb   = nn.Embedding(n_fids, emb_dim)
        self.state_emb = nn.Embedding(n_states, emb_dim)

        # Total input dimension after concatenating:
        #   CM rep (d_rep) + fid embedding + state embedding
        in_dim = d_rep + 2 * emb_dim

        # Initial projection (no residual here, we change dimension)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Stack of residual blocks in hidden_dim space
        self.backbone = nn.Sequential(
            ResBlock(hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, dropout=dropout),
        )

        # Final regression head
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        feats: torch.Tensor,      # (B, d_rep)  — scaled CM/PCA features
        fid_idx: torch.Tensor,    # (B,)        — fidelity indices
        state_idx: torch.Tensor,  # (B,)        — state indices
    ) -> torch.Tensor:
        # Look up embeddings
        fe = self.fid_emb(fid_idx)      # (B, emb_dim)
        se = self.state_emb(state_idx)  # (B, emb_dim)

        # Concatenate features + embeddings
        x = torch.cat([feats, fe, se], dim=-1)  # (B, d_rep + 2*emb_dim)

        # Project to hidden_dim, then apply residual blocks
        x = self.input_proj(x)   # (B, hidden_dim)
        x = self.backbone(x)     # (B, hidden_dim)

        # Final regression head
        out = self.out(x)        # (B, 1)

        return out.squeeze(-1)   # (B,)
