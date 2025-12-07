import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Simple residual block for MLPs:
      y = x + Dropout(GELU(BN(Linear(x))))
    """
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)