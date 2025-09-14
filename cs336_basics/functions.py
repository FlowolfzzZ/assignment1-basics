import torch
import math
from einops import einsum

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    # two parameters: a tensor and a dimension i
    max_value, _ = torch.max(x, dim=i, keepdim=True)
    new_x = x - max_value
    numerator = torch.exp(new_x)
    denominator = torch.sum(numerator, dim=i, keepdim=True)
    return numerator / denominator

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        QK.masked_fill_(~mask, float('-inf'))
    softmax_QK = softmax(QK / math.sqrt(d_k), -1)
    return einsum(softmax_QK, V, "... queries keys, ... keys d_v -> ... queries d_v")