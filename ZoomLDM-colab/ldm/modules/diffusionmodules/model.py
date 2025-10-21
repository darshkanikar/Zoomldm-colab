import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from typing import Optional, Any

from ldm.modules.attention import MemoryEfficientCrossAttention

# xformers check
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("No module 'xformers'. Proceeding without it.")


def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = nn.functional.pad(x, (0, 1, 0, 1))
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, 2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, temb=None):
        h = self.conv1(nonlinearity(self.norm1(x)))
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(self.norm(x)).reshape(B, C, H * W).permute(0, 2, 1)
        k = self.k(self.norm(x)).reshape(B, C, H * W)
        v = self.v(self.norm(x)).reshape(B, C, H * W)
        w = torch.bmm(q, k) * C ** -0.5
        w = nn.functional.softmax(w, dim=2)
        h = torch.bmm(v, w.permute(0, 2, 1)).reshape(B, C, H, W)
        return x + self.proj_out(h)


class MemoryEfficientAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        if not XFORMERS_IS_AVAILABLE:
            raise RuntimeError("xformers not available for MemoryEfficientAttnBlock")
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = [rearrange(t, "b c h w -> b (h w) c") for t in (self.q(self.norm(x)), self.k(self.norm(x)), self.v(self.norm(x)))]
        out = xformers.ops.memory_efficient_attention(q, k, v)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return x + self.proj_out(out)


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    if XFORMERS_IS_AVAILABLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unknown attn_type {attn_type}")
