import torch
import math
from einops import einsum
from collections.abc import Iterable

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

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    inputs_max, _ = torch.max(inputs, dim=-1, keepdim=True)
    inputs -= inputs_max
    loss = -torch.gather(inputs, dim=-1, index=torch.unsqueeze(targets, dim=-1)) + torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True))
    return torch.mean(loss)

def learning_rate_schedule(t: int, max_learning_rate: float, min_learning_rate: float, T_w: int, T_c: int):
    if t < T_w:
        return max_learning_rate * t / T_w
    elif t <= T_c:
        return min_learning_rate + (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (max_learning_rate - min_learning_rate) / 2
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    for param in parameters:
        norm = torch.norm(param.grad, p=2)
        if norm >= max_l2_norm:
            param.grad *= max_l2_norm / (norm + eps)