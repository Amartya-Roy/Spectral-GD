"""
Model definitions for Pretraining on FineWeb:
1. Pythia-160M (12 layers, model dimension 768, 12 heads, parallel residual, partial RoPE 25%, biases=True)
2. Modded-NanoGPT (11 layers, model dimension 768, 6 heads, parameter-free RMSNorm, RoPE, ReLU^2 MLP, biases=False)
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len: int, rotary_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos[None, None, :, :] + rotate_half(x) * sin[None, None, :, :]


# -----------------------------------------------------------------------------
# 1. Pythia-160M (12 layers, model dimension 768)
# -----------------------------------------------------------------------------
@dataclass
class Pythia160MConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    rotary_pct: float = 0.25
    bias: bool = True


class PythiaParallelBlock(nn.Module):
    def __init__(self, config: Pythia160MConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rotary_ndims = int(self.head_dim * config.rotary_pct)

        self.input_layernorm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.post_attention_layernorm = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.attn_out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.dense_h_to_4h = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.dense_4h_to_h = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def _attn(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        rd = self.rotary_ndims
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q = torch.cat([apply_rotary_pos_emb(q_rot, cos, sin), q_pass], dim=-1)
        k = torch.cat([apply_rotary_pos_emb(k_rot, cos, sin), k_pass], dim=-1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out_proj(y)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x + self._attn(self.input_layernorm(x), cos, sin) + self._mlp(self.post_attention_layernorm(x))


class Pythia160M(nn.Module):
    def __init__(self, config: Pythia160MConfig):
        super().__init__()
        self.config = config
        self.embed_in = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([PythiaParallelBlock(config) for _ in range(config.n_layer)])
        self.final_layer_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.embed_out = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        rotary_dim = int(head_dim * config.rotary_pct)
        cos, sin = build_rope_cache(config.block_size, rotary_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        x = self.embed_in(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_layer_norm(x)
        logits = self.embed_out(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# -----------------------------------------------------------------------------
# 2. Modded-NanoGPT (11 layers, model dimension 768)
# -----------------------------------------------------------------------------
@dataclass
class ModdedNanoGPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 11
    n_head: int = 6
    n_embd: int = 768


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class ModdedNanoGPTBlock(nn.Module):
    def __init__(self, config: ModdedNanoGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def _attn(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = rms_norm(q), rms_norm(k)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_proj(F.relu(self.c_fc(x)).square())

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(rms_norm(x), cos, sin)
        x = x + self._mlp(rms_norm(x))
        return x


class ModdedNanoGPT(nn.Module):
    def __init__(self, config: ModdedNanoGPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([ModdedNanoGPTBlock(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(config.block_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        x = rms_norm(self.wte(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = rms_norm(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
