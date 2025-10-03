import torch
import torch.nn as nn
import math
from einops import einsum
from collections.abc import Iterable
import numpy
import os
import typing
import numpy as np
from cs336_basics.Transformer import Transformer
from cs336_basics.Optimizer import AdamW
from cs336_basics.Tokenizer import Tokenizer

def softmax(x: torch.Tensor, i: int, temperature: float = 1.0) -> torch.Tensor:
    # two parameters: a tensor and a dimension i
    max_value, _ = torch.max(x, dim=i, keepdim=True)
    new_x = (x - max_value) / temperature
    numerator = torch.exp(new_x)
    denominator = torch.sum(numerator, dim=i, keepdim=True)
    return numerator / denominator

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    targets = targets.to(torch.long)
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
    input = []
    next_token = []
    sample_start = np.random.randint(low=0, high=len(x)-context_length, size=(batch_size,))
    for i in range(batch_size):
        start = sample_start[i]
        input.append(torch.tensor(x[start:start+context_length], device=device))
        next_token.append(torch.tensor(x[start+1:start+context_length+1], device=device))
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

def decoding(model: nn.Module, tokenizer: Tokenizer, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    special_token = "<|endoftext|>"
    special_token_id = torch.tensor(tokenizer.encode(special_token), device=model.device)
    x = torch.tensor(tokenizer.encode(prompt), device=model.device)
    y = []
    start = len(x)
    for t in range(start, max_tokens):
        logits = model(x)[-1, :]
        if temperature == 0:
            max_logit, _ = torch.max(logits, dim=-1, keepdim=True)
            probs = torch.where(logits >= max_logit, 1.0, 0.0)
        else:
            probs = softmax(logits, -1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs <= top_p
        sorted_mask[0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[sorted_indices] = sorted_mask  
        probs = torch.where(mask, probs, torch.tensor(0.0))

        denominator = torch.sum(probs, dim=-1, keepdim=True)
        probs /= denominator
        sample_token_id = torch.multinomial(probs, num_samples=1)
        y.append(int(sample_token_id))
        x = torch.cat((x, sample_token_id), dim=0)
        if int(sample_token_id) == int(special_token_id):
            break
    return tokenizer.decode(y)

def generate(model_path: str, vocab_path: str, merges_path: str, prompt: str, max_tokens: int, temperature: float, top_p: float, device: str = "cuda:0") -> str:
    model = Transformer(10000, 256, 512, 4, 16, 1344, 10000, device=device, dtype=torch.bfloat16)
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    load_checkpoint(model_path, model, optimizer)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    return decoding(model, tokenizer, prompt, max_tokens, temperature, top_p)

if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate("results/training_TinyStories/final_model_lr_3e-3_3e-4",
                              "results/bpe_tinystories_vocab.pkl",
                              "results/bpe_tinystories_merges.pkl",
                              prompt,
                              256, 1.0, 0.9, device="cuda:0")
    print(f"{prompt}{generated_text}")