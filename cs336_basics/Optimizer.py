from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        defaults = self.defaults
        for group in self.param_groups:
            lr = defaults["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data # Get the gradient of loss with respect to p.
                state = self.state[p] # Get state associated with p.
                
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p)) # Get the first moment vector.
                v = state.get("v", torch.zeros_like(p)) # Get the second moment vector.
                state["m"] = defaults["betas"][0] * m + (1 - defaults["betas"][0]) * grad # Update the first moment estimate.
                state["v"] = defaults["betas"][1] * v + (1 - defaults["betas"][1]) * (grad ** 2) # Update the first moment estimate.
                lr_t = lr * math.sqrt(1 - defaults["betas"][1] ** t) / (1 - defaults["betas"][0] ** t)
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + defaults["eps"])
                p.data -= lr * defaults["weight_decay"] * p.data
                state["t"] = t + 1 # Increment iteration number.
        return loss