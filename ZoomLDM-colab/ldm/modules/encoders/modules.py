import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder


# -----------------------------
# BASE ENCODERS
# -----------------------------
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x


# -----------------------------
# CLASS EMBEDDING
# -----------------------------
class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class", ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes + 1, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        key = key or self.key
        c = batch[key][:, None]  # Shape: [B,1]
        if self.ucg_rate > 0.0 and not disable_dropout:
            mask = 1.0 - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * (self.n_classes)  # unconditional class token
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device=None):
        device = device or "cuda"
        uc_class = self.n_classes  # extra class token for unconditional
        uc = torch.ones((bs,), device=device) * uc_class
        return {self.key: uc}


# -----------------------------
# EMBEDDING ViT2
# -----------------------------
class EmbeddingViT2(nn.Module):
    """
    ViT-based embedding module with magnitude token concatenation.
    """

    def __init__(
        self,
        feat_key="feat",
        mag_key="mag",
        input_channels=1024,
        hidden_channels=512,
        vit_mlp_dim=2048,
        output_channels=512,
        seq_length=64,
        mag_levels=8,
        num_layers=12,
        num_heads=8,
        p_uncond=0.0,
        ckpt_path=None,
        ignore_keys=None,
    ):
        super().__init__()
        ignore_keys = ignore_keys or []

        self.feat_key = feat_key
        self.mag_key = mag_key
        self.hidden_channels = hidden_channels
        self.p_uncond = p_uncond

        self.mag_embedding = nn.Embedding(mag_levels, hidden_channels)
        self.dim_reduce = nn.Linear(input_channels, hidden_channels)
        self.pad_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

        self.encoder = Encoder(
            seq_length=seq_length + 1,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_channels,
            mlp_dim=vit_mlp_dim,
            dropout=0,
            attention_dropout=0,
        )
        self.final_proj = nn.Linear(hidden_channels, output_channels)

    def process_input_batch(self, x):
        # Supports list or tensor input
        if isinstance(x, torch.Tensor):
            x = list(x)
        return torch.stack([self.process_single_input(item) for item in x])

    def process_single_input(self, x):
        # Ensure [C,H,W] -> [tokens, hidden_channels]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [C,H]
        c, h, w = x.shape
        n = h * w
        x = x.view(c, -1).transpose(0, 1)  # [tokens, C]
        x = self.dim_reduce(x)

        # Handle padding / center token
        if n < 64:
            pad_len = 64 - n
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            x = F.pad(x, (0, 0, pad_left, pad_right))
            mask = torch.ones(64, device=x.device)
            mask[pad_left:pad_left + n] = 0
            x = x * (1 - mask.unsqueeze(1)) + self.pad_token * mask.unsqueeze(1)
        return x  # [64, hidden_channels]

    def forward(self, batch):
        x = batch[self.feat_key]
        int_mag = batch[self.mag_key]

        x = self.process_input_batch(x)  # [B, 64, hidden_channels]
        mag_embed = self.mag_embedding(int_mag).unsqueeze(1)  # [B,1,H]
        x = torch.cat((mag_embed, x), dim=1)  # [B, 65, H]

        x = self.encoder(x)
        x = self.final_proj(x)

        # apply unconditional masking
        if self.p_uncond > 0.0:
            mask = 1.0 - torch.bernoulli(torch.ones(x.size(0), device=x.device) * self.p_uncond)
            mask = mask[:, None, None]
            x = x * mask
        return x

    def encode(self, batch):
        return self.forward(batch)


# -----------------------------
# EMBEDDING ViT2_5: LayerNorm at the end
# -----------------------------
class EmbeddingViT2_5(EmbeddingViT2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm = nn.LayerNorm(self.hidden_channels)

    def forward(self, batch):
        x = super().forward(batch)
        x = self.layer_norm(x)
        return x

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location="cpu")["state_dict"]
        sd_cond_stage = {k.replace("cond_stage_model.", ""): v for k, v in sd.items() if "cond_stage_model" in k}
        self.load_state_dict(sd_cond_stage, strict=True)
        print(f"Restored from {path}")
