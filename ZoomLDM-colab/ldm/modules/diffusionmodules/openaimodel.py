import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

# -----------------------------
# STUBS / UTILITY FUNCTIONS
# -----------------------------
def conv_nd(dims, in_ch, out_ch, kernel_size, stride=1, padding=0):
    if dims == 1:
        return nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
    elif dims == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
    elif dims == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)

def linear(in_ch, out_ch):
    return nn.Linear(in_ch, out_ch)

def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def avg_pool_nd(dims, kernel_size, stride):
    if dims == 1:
        return nn.AvgPool1d(kernel_size, stride=stride)
    elif dims == 2:
        return nn.AvgPool2d(kernel_size, stride=stride)
    elif dims == 3:
        return nn.AvgPool3d(kernel_size, stride=stride)

def checkpoint(func, args, params, use_checkpoint):
    # Simple pass-through; gradient checkpointing skipped for Colab
    return func(*args)

def timestep_embedding(timesteps, dim, repeat_only=False):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = th.exp(th.arange(half, dtype=th.float32) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = th.cat([emb.sin(), emb.cos()], dim=1)
    if dim % 2 == 1:
        emb = th.cat([emb, th.zeros(timesteps.shape[0], 1)], dim=1)
    return emb

def convert_module_to_f16(module):
    module.half()

def convert_module_to_f32(module):
    module.float()

def exists(val):
    return val is not None

# -----------------------------
# SPATIAL TRANSFORMER STUB
# -----------------------------
class SpatialTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, context=None):
        return x

# -----------------------------
# CORE MODULES
# -----------------------------
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads_channels: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3]*2, x.shape[4]*2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1,2,2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2*self.out_channels if use_scale_shift_norm else self.out_channels)
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

# -----------------------------
# QKV Attention
# -----------------------------
class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q*scale).view(bs*self.n_heads, ch, length),
            (k*scale).view(bs*self.n_heads, ch, length)
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs*self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

# -----------------------------
# UNetModel
# -----------------------------
class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels,
                 num_res_blocks, attention_resolutions, channel_mult=(1,2,4), use_checkpoint=False):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks] if isinstance(num_res_blocks,int) else num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.use_checkpoint = use_checkpoint

        time_embed_dim = model_channels*4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, 3, padding=1))
        ])
        self.middle_block = TimestepEmbedSequential(
            ResBlock(model_channels, time_embed_dim, 0.0)
        )
        self.output_blocks = nn.ModuleList([
            TimestepEmbedSequential(ResBlock(model_channels*channel_mult[-1], time_embed_dim, 0.0))
        ])
        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(2, model_channels, out_channels, 3, padding=1))
        )

    def forward(self, x, timesteps=None, context=None):
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        h = x
        for block in self.input_blocks:
            h = block(h, emb, context)
        h = self.middle_block(h, emb, context)
        for block in self.output_blocks:
            h = block(h, emb, context)
        return self.out(h)

# -----------------------------
# TEST RUN
# -----------------------------
if __name__ == "__main__":
    model = UNetModel(
        image_size=64,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[4,8]
    )

    x = th.randn(1, 3, 64, 64)
    t = th.tensor([10])
    out = model(x, timesteps=t)
    print("Output shape:", out.shape)
