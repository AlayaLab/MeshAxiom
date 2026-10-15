"""Transformer building blocks shared by the two MeshAxiom diffusion models.

Design follows DiT / PixArt: adaLN-Zero self-attention + zero-gated
cross-attention to the condition tokens, plus 3D RoPE on the spatial position
of every token (octree node centre or vertex position).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of a continuous time in [0, 1] (scaled by 1000)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    a = t.float().reshape(-1, 1) * 1000.0 * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.reshape(*t.shape, dim)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype))


# --------------------------------------------------------------------------- #
# 3D rotary position embedding
# --------------------------------------------------------------------------- #
def axis_dims(head_dim: int) -> list[int]:
    """Split ``head_dim`` into three even per-axis budgets."""
    base = 2 * (head_dim // 6)
    return [base, base, head_dim - 2 * base]


class RoPE3D(nn.Module):
    """Axial rotary embedding over an (x, y, z) position per token."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.dims = axis_dims(head_dim)
        assert all(d % 2 == 0 for d in self.dims), self.dims
        self.head_dim = head_dim
        for i, d in enumerate(self.dims):
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            self.register_buffer(f"inv_{i}", inv, persistent=False)

    def cos_sin(self, pos: torch.Tensor):
        """``pos``: (B, N, 3) float positions -> cos/sin of shape (B, N, head_dim).

        Angles are accumulated in float64: positions run up to 2**D (=512 finest-grid
        units) and the highest-frequency band has inv_freq = 1, so a float32 angle of
        ~512 rad would lose 2-3 significant digits and break the relative-position
        property that attention relies on.
        """
        outs = []
        for i in range(3):
            inv = getattr(self, f"inv_{i}").double()
            ang = pos[..., i : i + 1].double() * inv.reshape(*([1] * (pos.dim() - 1)), -1)
            outs.append(torch.repeat_interleave(ang, 2, dim=-1))
        ang = torch.cat(outs, dim=-1)
        return ang.cos().float(), ang.sin().float()

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``x``: (B, H, N, D); ``cos``/``sin``: (B, N, D).

        The rotation is applied in float32 and cast back: bf16 carries ~3 decimal
        digits, which is not enough to keep the rotation faithful at large positions.
        """
        xf = x.float()
        cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return (xf * cos + rot * sin).to(x.dtype)


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #
def _sdpa_padded(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _sdpa_varlen_flash(q, k, v, mask=None):
    """V186 (09-28): exact softmax attention through PyTorch's built-in varlen FlashAttention kernel for padded batches.

    The production path hands a (B,1,1,N) bool padding mask to SDPA, which excludes the FlashAttention backend and runs the
    memory-efficient kernel over the padded (B,H,N,D) tensors.  Here the valid tokens are packed to (total,H,D) with cumulative
    sequence lengths and attended with the flash kernel; padded query rows come back as zeros (they are never read).  Same
    mathematics, only bf16 rounding differs (the topology DiT admitted the identical construction as V92).  Used only when a mask
    is given and the inputs are bf16/fp16; everything else (no mask = sampling, fp32) stays on the production path."""
    if mask is None or q.dtype not in (torch.bfloat16, torch.float16):
        return _sdpa_padded(q, k, v, mask)
    b, heads, n, d = q.shape
    valid = mask.reshape(b, n)
    lengths = valid.sum(1)
    cu = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
    longest = int(lengths.max())
    packed = [x.permute(0, 2, 1, 3)[valid] for x in (q, k, v)]
    out = torch.ops.aten._flash_attention_forward(packed[0], packed[1], packed[2], cu, cu, longest, longest, 0.0, False, False,
                                                  scale=d ** -0.5)[0]
    full = out.new_zeros(b, n, heads, d)
    full[valid] = out
    return full.permute(0, 2, 1, 3)


import os as _os
_sdpa = _sdpa_varlen_flash if _os.environ.get("MESHAXIOM_VARLEN_FLASH") == "1" else _sdpa_padded


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x, rope=None, mask=None):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        if rope is not None:
            q = RoPE3D.apply(q, *rope)
            k = RoPE3D.apply(k, *rope)
        x = _sdpa(q, k, v, mask)
        return self.proj(x.transpose(1, 2).reshape(b, n, -1))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, ctx_dim: int, qk_norm: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.hd = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x, ctx, mask=None):
        b, n, _ = x.shape
        m = ctx.shape[1]
        q = self.q(x).reshape(b, n, self.heads, self.hd).transpose(1, 2)
        k, v = self.kv(ctx).reshape(b, m, 2, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        x = _sdpa(self.qn(q), self.kn(k), v, mask)
        return self.proj(x.transpose(1, 2).reshape(b, n, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(dim * mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
class DiTBlock(nn.Module):
    """adaLN-Zero self-attention + zero-gated cross-attention + MLP."""

    def __init__(self, dim: int, heads: int, ctx_dim: int | None = None, mlp_mult: float = 4.0,
                 qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads, qk_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, mlp_mult)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)
        self.use_ctx = ctx_dim is not None
        if self.use_ctx:
            self.norm_ctx = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross = CrossAttention(dim, heads, ctx_dim, qk_norm)
            self.ctx_gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x, c, rope=None, mask=None, ctx=None, ctx_mask=None, side=None, qpe=None):
        """``c``: (B, 1|N, dim) conditioning vector for adaLN.  ``side`` (V181 multi-view): (ctx2 (B,M2,D), valid mask (B,M2) bool,
        has (B,) float, gate (dim,)) -> x += gate * cross(norm_ctx(x), ctx2) * has, through the same cross module as the image context.
        ``qpe`` (V182 posenc182): node-position Fourier features; the block's zero-initialised ``q_pos`` adds them to the query input
        of the image cross-attention (the side branch keeps position-free queries)."""
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h, rope=rope, mask=mask)
        if self.use_ctx and ctx is not None:
            xn = xq = self.norm_ctx(x)
            if qpe is not None and getattr(self, "q_pos", None) is not None:   # V182: absolute node position for the image queries
                xq = xn + self.q_pos(qpe).to(xn.dtype)
            x = x + self.ctx_gate * self.cross(xq, ctx, ctx_mask)
            if side is not None:   # V181 side branch unchanged (position-free queries/tokens: a shared code would impose the main-view geometry)
                ctx2, m2, has, g = side
                x = x + g * self.cross(xn, ctx2, m2[:, None, None, :]) * has[:, None, None].to(x.dtype)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.proj(self.norm(x) * (1 + scale) + shift)


class TransformerBlock(nn.Module):
    """Plain pre-norm transformer block (VAE encoder/decoder, condition encoder)."""

    def __init__(self, dim: int, heads: int, mlp_mult: float = 4.0, qk_norm: bool = True,
                 use_mlp: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, heads, qk_norm)
        # Ablation T1: the paper calls the VAE decoder "pure attention" and its ~100M
        # parameter count only adds up without MLPs; ``use_mlp=False`` is that reading.
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) if use_mlp else None
        self.mlp = MLP(dim, mlp_mult) if use_mlp else None

    def forward(self, x, rope=None, mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, mask=mask)
        return x + self.mlp(self.norm2(x)) if self.mlp is not None else x


def padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) lengths -> (B, 1, 1, max_len) bool mask usable by SDPA."""
    ar = torch.arange(max_len, device=lengths.device)
    return (ar[None, :] < lengths[:, None])[:, None, None, :]
