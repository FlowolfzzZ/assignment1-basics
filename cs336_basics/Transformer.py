import torch
import torch.nn as nn
import math
from einops import einsum, repeat, rearrange

def softmax(x: torch.Tensor, i: int, temperature: float = 1.0) -> torch.Tensor:
    # two parameters: a tensor and a dimension i
    max_value, _ = torch.max(x, dim=i, keepdim=True)
    new_x = (x - max_value) / temperature
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

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

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
        indices = token_ids.to(torch.long)
        indices = indices.to(self.device)
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
        x = x.to(self.dtype)
        rms = torch.unsqueeze(torch.sqrt(torch.sum(x ** 2, dim=-1) / self.d_model + self.eps), dim=-1)
        result = torch.mul(x, self.weight) / rms
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w1 = self.w1(x)
        silu = torch.mul(x_w1, torch.sigmoid(x_w1))
        return self.w2(torch.mul(self.w3(x), silu))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        rotation_matrix = torch.zeros(max_seq_len, d_k, d_k, device=device, dtype=dtype)
        for i in range(max_seq_len):
            for k in range(1, d_k // 2 + 1):
                theta_i_k = i / (theta ** ((2*k-2) / d_k))
                rotation_matrix[i, 2*k-2:2*k, 2*k-2:2*k] = torch.tensor([[math.cos(theta_i_k), math.sin(theta_i_k)],
                                                                         [-math.sin(theta_i_k), math.cos(theta_i_k)]])
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=self.device)
        token_position_indices = token_positions.to(torch.long)
        token_position_indices = token_position_indices.to(self.device)
        token_rotation_matrix = self.rotation_matrix[token_position_indices]
        return einsum(x, token_rotation_matrix, "... seq_len d_k, ... seq_len d_k d_k2 -> ... seq_len d_k2")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int = None,
                 theta: float = None,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.d_k = self.d_v = self.d_model // self.num_heads
        self.RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device, dtype=dtype) if max_seq_len is not None else None
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        qkv_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        Q, K, V = (x @ qkv_weight.T).chunk(3, -1)
        Q = rearrange(Q, "... seq_len (num_heads d) -> ... num_heads seq_len d", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d) -> ... num_heads seq_len d", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d) -> ... num_heads seq_len d", num_heads=self.num_heads)
        seq_len = Q.shape[-2]
        if self.RoPE is not None:
            Q = self.RoPE(Q, token_positions)
            K = self.RoPE(K, token_positions)
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        attn = rearrange(scaled_dot_product_attention(Q, K, V, mask), "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.output_proj(attn)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None,
                 device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        blocks = [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for i in range(num_layers)]
        self.layers = nn.ModuleList(blocks)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, in_indices: torch.Tensor):
        x = self.token_embeddings(in_indices)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return self.lm_head(self.ln_final(x))