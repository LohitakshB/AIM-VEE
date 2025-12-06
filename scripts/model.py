import torch

class Model(torch.nn.Module):
    def __init__(self, d_rep, n_fids, n_states,
                 hidden_dim=256, emb_dim=16):
        super().__init__()

        self.fid_emb   = torch.nn.Embedding(n_fids, emb_dim)
        self.state_emb = torch.nn.Embedding(n_states, emb_dim)

        in_dim = d_rep + 2 * emb_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats, fid_idx, state_idx):
        # feats: (B, d_rep) already scaled
        fe = self.fid_emb(fid_idx)      # (B, emb_dim)
        se = self.state_emb(state_idx)  # (B, emb_dim)
        x = torch.cat([feats, fe, se], dim=-1)
        out = self.mlp(x)
        return out.squeeze(-1)


