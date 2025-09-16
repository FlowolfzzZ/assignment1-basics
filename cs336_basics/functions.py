import torch
import math
from einops import einsum
from collections.abc import Iterable
import numpy
import os
import typing

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
    total = 0
    for param in parameters:
        if param.grad is not None:
            total += torch.norm(param.grad, p=2) ** 2
    total = math.sqrt(total)
    if total >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm / (total + eps)

def get_batch(x: numpy.typing.NDArray, batch_size: int, context_length: int, device: str):
    x = torch.Tensor(x, device=device)
    input = []
    next_token = []
    sample_start = torch.randint(low=0, high=len(x)-context_length, size=(batch_size,))
    for i in range(batch_size):
        start = sample_start[i]
        input.append(x[start:start+context_length])
        next_token.append(x[start+1:start+context_length+1])
    return (torch.stack(input, dim=0), torch.stack(next_token, dim=0))

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    state_dict = {}
    state_dict["model"] = model.state_dict()
    state_dict["optimizer"] = optimizer.state_dict()
    state_dict["iteration"] = iteration
    torch.save(state_dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]