import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        weight = torch.empty(in_features, out_features)
        self.weight = nn.Parameter(weight)
        sigma = torch.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight