import torch
import torch.nn as nn
import torch.nn.functional as F


# Bayes-by-Backprop Linear layer
class BayesLinear(nn.Module):
    """
    Mean-field variational Bayesian Linear layer (Bayes by Backprop).
    W ~ N(mu, sigma^2), sigma = softplus(rho)
    Forward returns (y, kl)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        init_mu_std: float = 0.02,
        init_rho: float = -5.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.use_bias = bias

        # Variational params for weights
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, init_mu_std))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features).fill_(init_rho))

        # Variational params for bias
        if self.use_bias:
            self.b_mu = nn.Parameter(torch.empty(out_features).normal_(0, init_mu_std))
            self.b_rho = nn.Parameter(torch.empty(out_features).fill_(init_rho))
        else:
            self.register_parameter("b_mu", None)
            self.register_parameter("b_rho", None)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-8

    @staticmethod
    def _kl_normal(mu_q, sigma_q, mu_p, sigma_p) -> torch.Tensor:
        # KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)), summed
        return (
            torch.log(sigma_p / sigma_q)
            + (sigma_q**2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p**2)
            - 0.5
        ).sum()

    def forward(self, x: torch.Tensor, sample: bool = True):
        w_sigma = self._sigma(self.w_rho)
        if self.use_bias:
            b_sigma = self._sigma(self.b_rho)

        # Sample via reparameterization during training (or when explicitly sampling)
        if sample:
            eps_w = torch.randn_like(self.w_mu)
            w = self.w_mu + w_sigma * eps_w
            if self.use_bias:
                eps_b = torch.randn_like(self.b_mu)
                b = self.b_mu + b_sigma * eps_b
            else:
                b = None
        else:
            # Deterministic: posterior mean
            w = self.w_mu
            b = self.b_mu if self.use_bias else None

        y = F.linear(x, w, b)

        # KL to prior N(0, prior_sigma^2)
        prior_mu = 0.0
        prior_sigma = self.prior_sigma
        kl = self._kl_normal(self.w_mu, w_sigma, prior_mu, prior_sigma)
        if self.use_bias:
            kl = kl + self._kl_normal(self.b_mu, b_sigma, prior_mu, prior_sigma)

        return y, kl



# Bayesian ResBlock using BayesLinear
class ResBlock(nn.Module):
    """
    Residual block but Bayesian (BayesLinear inside).
    Returns (out, kl_block).
    """
    def __init__(self, dim: int, dropout: float = 0.2, prior_sigma: float = 1.0):
        super().__init__()
        self.fc1 = BayesLinear(dim, dim, prior_sigma=prior_sigma)
        self.fc2 = BayesLinear(dim, dim, prior_sigma=prior_sigma)

        # LayerNorm is more stable than BatchNorm with stochastic weights
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sample: bool = True):
        residual = x

        h, kl1 = self.fc1(x, sample=sample)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)

        h, kl2 = self.fc2(h, sample=sample)
        h = self.norm2(h)
        h = self.drop(h)

        out = self.act(h + residual)
        return out, (kl1 + kl2)



# VEE Predictor Model
class vee_predictor(nn.Module):
    def __init__(
        self,
        d_rep: int,
        n_fids: int,
        n_states: int,
        hidden_dim: int = 128,
        emb_dim: int = 32,
        dropout: float = 0.2,
        prior_sigma: float = 1.0,
    ):
        super().__init__()

        self.fid_emb   = nn.Embedding(n_fids, emb_dim)
        self.state_emb = nn.Embedding(n_states, emb_dim)

        in_dim = d_rep + 2 * emb_dim

        # Bayesian input projection
        self.input_fc = BayesLinear(in_dim, hidden_dim, prior_sigma=prior_sigma)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_act = nn.GELU()

        self.backbone = nn.ModuleList([
            ResBlock(hidden_dim, dropout=dropout, prior_sigma=prior_sigma),
            ResBlock(hidden_dim, dropout=dropout, prior_sigma=prior_sigma),
            ResBlock(hidden_dim, dropout=dropout, prior_sigma=prior_sigma),
        ])

        # Bayesian output head
        self.out_fc = BayesLinear(hidden_dim, 1, prior_sigma=prior_sigma)

    def forward(
        self,
        feats: torch.Tensor,      # (B, d_rep)
        fid_idx: torch.Tensor,    # (B,)
        state_idx: torch.Tensor,  # (B,)
        sample: bool = True,
        return_kl: bool = False,
    ):
        fe = self.fid_emb(fid_idx)
        se = self.state_emb(state_idx)
        x = torch.cat([feats, fe, se], dim=-1)

        kl_total = 0.0

        x, kl = self.input_fc(x, sample=sample)
        kl_total = kl_total + kl
        x = self.input_norm(x)
        x = self.input_act(x)

        for block in self.backbone:
            x, klb = block(x, sample=sample)
            kl_total = kl_total + klb

        out, kl_out = self.out_fc(x, sample=sample)
        kl_total = kl_total + kl_out

        out = out.squeeze(-1)

        if return_kl:
            return out, kl_total
        return out
