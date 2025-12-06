import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, d_rep, n_fids, n_states,
                 hidden_dim=512, emb_dim=32, dropout=0.2):
        """
        Feed-forward regression model with:
          - embeddings for fidelity and state
          - deeper MLP
          - BatchNorm + GELU + Dropout for better generalization
        """
        super().__init__()

        self.fid_emb   = nn.Embedding(n_fids, emb_dim)
        self.state_emb = nn.Embedding(n_states, emb_dim)

        in_dim = d_rep + 2 * emb_dim  # CM rep + fid emb + state emb

        def block(in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # 3 hidden blocks + final linear head
        self.mlp = nn.Sequential(
            block(in_dim,      hidden_dim),
            block(hidden_dim,  hidden_dim),
            block(hidden_dim,  hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, fid_idx, state_idx):
        """
        feats: (B, d_rep)  — already scaled with StandardScaler
        fid_idx: (B,)      — integer fidelity indices
        state_idx: (B,)    — integer state indices
        """
        fe = self.fid_emb(fid_idx)      # (B, emb_dim)
        se = self.state_emb(state_idx)  # (B, emb_dim)

        x = torch.cat([feats, fe, se], dim=-1)  # (B, d_rep + 2*emb_dim)
        out = self.mlp(x)                       # (B, 1)

        return out.squeeze(-1)                  # (B,)
