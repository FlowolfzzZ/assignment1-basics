import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        weight = torch.empty(in_features, out_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, d_model: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        weight = torch.empty(num_embeddings, d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        std = 1.0
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        indices = torch.LongTensor(token_ids)
        return self.weight[indices]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        weight = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.unsqueeze(torch.sqrt(torch.sum(x ** 2, dim=-1) / self.d_model + self.eps), dim=-1)
        result = torch.mul(x, self.weight) / rms
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = int((d_model * 8 / 3) // 64 * 64)
        self.w1 = nn.Parameter(torch.empty(d_model, self.d_ff, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.d_ff, d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_model, self.d_ff, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w1 = x @ self.w1
        silu = torch.mul(x_w1, torch.sigmoid(x_w1))
        return torch.mul(x @ self.w3, silu) @ self.w2