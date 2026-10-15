"""Condition encoders for both diffusion stages (PAPER.txt:315-318, 500-510).

Point clouds go through a jointly trained VecSet encoder (8 layers, width 2048,
8192 points + normals -> 1024 tokens); images through a frozen DINOv3.  A
learnable null condition provides the unconditional branch for classifier-free
guidance (the paper never mentions CFG, so ``drop_prob`` defaults to 0).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from meshaxiom_core.models.fps import sample_query_indices
from meshaxiom_core.models.layers import CrossAttention, MLP, TransformerBlock

# NOTE(deviation): the paper does not give the VecSet input featurisation.  We
# use the usual axis-aligned Fourier features (8 octaves, input concatenated)
# plus the raw normals, as in 3DShape2VecSet / CLAY.
NUM_FREQS = 8


def fourier_embed(x: torch.Tensor, num_freqs: int = NUM_FREQS) -> torch.Tensor:
    """(..., C) -> (..., C * (1 + 2 * num_freqs)), octave-spaced sin/cos."""
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    a = x[..., None] * freqs
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], dim=-1)


def fourier_dim(channels: int = 3, num_freqs: int = NUM_FREQS) -> int:
    return channels * (1 + 2 * num_freqs)


class VecSetEncoder(nn.Module):
    """PAPER.txt:503-510 — VecSet, 8 layers, width 2048, 8192 points + normals -> 1024 tokens."""

    def __init__(self, dim=2048, depth=8, heads=16, num_latents=1024, mlp_mult=4.0):
        super().__init__()
        self.dim, self.num_latents = dim, num_latents
        self.fps_method, self.fps_start = "fps", "farthest"
        in_dim = fourier_dim(3) + 3  # Fourier(points) ++ raw normals
        self.in_proj = nn.Linear(in_dim, dim)
        self.q_proj = nn.Linear(in_dim, dim)
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.cross = CrossAttention(dim, heads, dim)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_mult)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_mult) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """``points``/``normals`` (B, P, 3) in [-1, 1] -> (B, num_latents, dim)."""
        feats = torch.cat([fourier_embed(points), normals], dim=-1)
        idx = sample_query_indices(points, self.num_latents, self.fps_method, self.fps_start)
        q = feats.gather(1, idx[..., None].expand(-1, -1, feats.shape[-1]))
        kv = self.norm_kv(self.in_proj(feats))
        x = self.q_proj(q)
        x = x + self.cross(self.norm_q(x), kv)
        x = x + self.mlp(self.norm_mlp(x))
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


class NullCondition(nn.Module):
    """Learnable null tokens (classifier-free guidance)."""

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

    def forward(self, batch: int) -> torch.Tensor:
        return self.tokens[None].expand(batch, -1, -1)


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 patch tokens (PAPER.txt:508-509), linearly projected to ``dim``."""

    def __init__(self, model_id: str, dim: int, image_size: int | None = None):
        super().__init__()
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_id)
        except Exception as e:  # never silently fall back (SPEC §3)
            raise RuntimeError(
                f"image conditioning needs DINOv3 weights for '{model_id}': {type(e).__name__}: {e}"
            ) from e
        self.backbone.eval().requires_grad_(False)
        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise RuntimeError(f"cannot read hidden_size from {model_id} config")
        pretrained_size = getattr(self.backbone.config, "image_size", None)
        patch_size = getattr(self.backbone.config, "patch_size", None)
        if not pretrained_size or not patch_size or int(pretrained_size) % int(patch_size):
            raise RuntimeError(f"cannot derive DINOv3 patch-token count from {model_id}")
        input_size = int(image_size or pretrained_size)
        if input_size % int(patch_size):
            raise ValueError(f"condition.image_size={input_size} is not divisible by "
                             f"DINOv3 patch_size={patch_size}")
        self.image_size = input_size
        self.num_tokens = (input_size // int(patch_size)) ** 2
        self.proj = nn.Linear(hidden, dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image`` (B, 3, H, W), already normalised by the DINOv3 processor."""
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"DINOv3 expected {self.image_size}x{self.image_size} pixels, "
                             f"got {tuple(image.shape[-2:])}")
        with torch.no_grad():
            out = self.backbone(pixel_values=image).last_hidden_state
        n_prefix = 1 + getattr(self.backbone.config, "num_register_tokens", 0)
        return self.proj(out[:, n_prefix:].to(self.proj.weight.dtype))


class ConditionEncoder(nn.Module):
    """Modality encoder + null tokens; always returns ``(B, M, ctx_dim)``.

    ``drop`` is a bool ``(B,)`` mask replacing the condition of those samples by
    the null tokens; when ``None`` it is drawn with ``drop_prob`` in training mode.
    """

    def __init__(self, kind: str, encoder: nn.Module | None, num_tokens: int, dim: int,
                 drop_prob: float = 0.0, camera_token: bool = False, lod_token: bool = False,
                 image_encoder: nn.Module | None = None, image_ratio: float = 0.5):
        super().__init__()
        self.kind, self.dim, self.num_tokens, self.drop_prob = kind, dim, num_tokens, drop_prob
        self.encoder = encoder
        # kind == "multi": PAPER.txt:500-512 describes ONE DiT "conditioned on point
        # clouds and images" with both encoders in the same breath -- the faithful
        # reading is joint training, one modality per sample.  ``encoder`` is then the
        # VecSet and ``image_encoder`` the DINOv3 tower.
        self.image_encoder = image_encoder
        # multi only: probability a training sample is conditioned on the image rather
        # than the point cloud.  The paper is silent; 0.5 is the neutral default.
        self.image_ratio = float(image_ratio)
        # Ablation I7: one extra context token carrying the camera pose (unit direction,
        # radius, fov -- a full extrinsic here, since every render looks at the origin).
        # ``num_tokens`` already counts it; the null tokens therefore cover it too.
        self.camera_mlp = nn.Sequential(
            nn.Linear(5, dim), nn.SiLU(), nn.Linear(dim, dim)) if camera_token else None
        # Paper-gap M2 (Fig. 9): a level-of-detail token carrying the target face count,
        # log-normalised to [0, ~1] over the paper's 500-20k range.  Works for any modality.
        self.lod_mlp = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)) if lod_token else None
        self.null = NullCondition(num_tokens, dim)
        if kind != "none" and drop_prob <= 0:
            # With no condition dropout the null tokens never appear in the loss, and DDP
            # refuses to run with a parameter that gets no gradient.  They carry no signal
            # in that regime anyway (CFG is meaningless without dropout during training),
            # so freezing them is a no-op semantically and avoids find_unused_parameters.
            self.null.tokens.requires_grad_(False)

    def forward(self, points=None, normals=None, image=None, camera=None, lod=None,
                drop=None) -> torch.Tensor:
        if self.kind == "none":
            ref = points if points is not None else image
            assert ref is not None, "condition type 'none' needs a tensor to size the batch"
            return self.null(ref.shape[0])
        if self.kind == "multi":
            have_p, have_i = points is not None, image is not None
            if have_p and have_i:
                # Joint training: both towers run and a per-sample coin selects the
                # modality.  Computing both keeps every parameter in the graph (zero
                # gradient for the unselected side), which DDP requires; the coin is
                # rank-local exactly like the CFG drop mask.
                ctx_p = self.encoder(points, normals)
                ctx_i = self.image_encoder(image).to(ctx_p.dtype)
                if ctx_p.shape[1] != ctx_i.shape[1]:
                    raise ValueError(f"multi conditioning needs equal token counts, got "
                                     f"pointcloud {ctx_p.shape[1]} vs image {ctx_i.shape[1]}")
                coin = torch.rand(ctx_p.shape[0], device=ctx_p.device) < self.image_ratio
                ctx = torch.where(coin[:, None, None], ctx_i, ctx_p)
            elif have_p:
                ctx = self.encoder(points, normals)
            elif have_i:
                ctx = self.image_encoder(image)
            else:
                raise ValueError("multi conditioning received neither points nor image")
        elif self.kind == "pointcloud":
            ctx = self.encoder(points, normals)
        else:
            ctx = self.encoder(image)
            if self.camera_mlp is not None:
                if camera is None:
                    raise ValueError("condition.camera_token=true but no camera pose was given")
                token = self.camera_mlp(camera.to(ctx.dtype))
                ctx = torch.cat([ctx, token[:, None, :]], dim=1)
        if self.lod_mlp is not None:
            if lod is None:
                raise ValueError("condition.lod_token=true but no target face count was given")
            feat = torch.log1p(lod.to(ctx.dtype).clamp_min(0.0)) / 9.903488  # log1p(20000)
            ctx = torch.cat([ctx, self.lod_mlp(feat[:, None, None])[:, 0][:, None, :]], dim=1)
        if drop is None and self.training and self.drop_prob > 0:
            drop = torch.rand(ctx.shape[0], device=ctx.device) < self.drop_prob
        # Always evaluate the null branch when a dropout mask was supplied, even
        # when this rank happened to draw an all-False mask.  ``torch.where`` then
        # gives ``null.tokens`` a real (zero-valued) gradient instead of leaving the
        # parameter unused.  That distinction matters under DDP: different ranks
        # commonly draw different all-False/non-empty masks for small local batches.
        if drop is not None:
            assert self.num_tokens == ctx.shape[1], \
                f"condition.num_tokens {self.num_tokens} != encoder tokens {ctx.shape[1]}"
            ctx = torch.where(drop[:, None, None], self.null(ctx.shape[0]).to(ctx.dtype), ctx)
        return ctx

    def null_ctx(self, batch: int) -> torch.Tensor:
        return self.null(batch)


def build_condition_encoder(cfg) -> tuple[nn.Module, int]:
    """``cfg["condition"]`` -> (encoder, ctx_dim).  Type: pointcloud | image | none."""
    c = dict(cfg["condition"])
    kind = c.get("type", "pointcloud")
    dim = int(c.get("dim", 2048))
    lod_token = bool(c.get("lod_token", False))
    if kind == "multi":
        enc = VecSetEncoder(dim=dim, depth=int(c.get("depth", 8)), heads=int(c.get("heads", 16)),
                            num_latents=int(c.get("num_latents", 1024)),
                            mlp_mult=float(c.get("mlp_mult", 4.0)))
        enc.fps_method = c.get("fps", "fps")
        img = DINOv3Encoder(c.get("model_id", "facebook/dinov3-vitl16-pretrain-lvd1689m"), dim,
                            int(c.get("image_size", 512)))
        if img.num_tokens != enc.num_latents:
            raise ValueError(f"multi conditioning needs equal token counts: VecSet "
                             f"{enc.num_latents} vs DINOv3 {img.num_tokens} (image_size)")
        num_tokens = enc.num_latents + int(lod_token)
        return ConditionEncoder(kind, enc, num_tokens, dim, float(c.get("drop_prob", 0.0)),
                                lod_token=lod_token, image_encoder=img,
                                image_ratio=float(c.get("image_ratio", 0.5))), dim
    if kind == "pointcloud":
        enc = VecSetEncoder(dim=dim, depth=int(c.get("depth", 8)), heads=int(c.get("heads", 16)),
                            num_latents=int(c.get("num_latents", 1024)),
                            mlp_mult=float(c.get("mlp_mult", 4.0)))
        enc.fps_method = c.get("fps", "fps")
        num_tokens = enc.num_latents + int(lod_token)
    elif kind == "image":
        enc = DINOv3Encoder(c.get("model_id", "facebook/dinov3-vitl16-pretrain-lvd1689m"), dim,
                            int(c.get("image_size", 512)))
        camera_token = bool(c.get("camera_token", False))
        produced = enc.num_tokens + int(camera_token) + int(lod_token)
        num_tokens = int(c.get("num_tokens", produced))
        if num_tokens != produced:
            extras = ([" plus 1 camera token"] if camera_token else []) + \
                     ([" plus 1 LOD token"] if lod_token else [])
            raise ValueError(f"condition.num_tokens={num_tokens}, but the configured DINOv3 "
                             f"produces {enc.num_tokens} patch tokens"
                             + "".join(extras)
                             + f" = {produced}; num_tokens is COUPLED to camera_token/"
                               f"lod_token and all must move together")
    elif kind == "none":
        enc, num_tokens = None, int(c.get("num_tokens", 64))
    else:
        raise ValueError(f"unknown condition type {kind!r}")
    return ConditionEncoder(kind, enc, num_tokens, dim, float(c.get("drop_prob", 0.0)),
                            camera_token=(kind == "image" and bool(c.get("camera_token", False))),
                            lod_token=lod_token), dim
