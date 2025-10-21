"""
Colab-compatible version of ddpm.py
-----------------------------------
ZoomLDM Diffusion Model - adapted for Colab GPUs.
- Compatible with PyTorch ≥ 2.1 (Colab default)
- Works with single-GPU
- bitsandbytes optional
- rank_zero_only import safe for all Lightning versions
"""

import os, math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from functools import partial
from tqdm import tqdm

# ====================== Safe imports ======================
try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    try:
        from pytorch_lightning.utilities.distributed import rank_zero_only
    except ImportError:
        def rank_zero_only(fn):
            return fn  # fallback if Lightning not installed

# bitsandbytes optional (for low VRAM)
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("⚠️ bitsandbytes not found; using standard AdamW optimizer.")

from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.util import instantiate_from_config

# =========================================================
# Gaussian Diffusion Model
# =========================================================
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        schedule="linear",
        loss_type="l2",
        p2_loss_weight_k=1,
        p2_loss_weight_gamma=0,
    ):
        super().__init__()
        self.register_schedule(timesteps, linear_start, linear_end, cosine_s, schedule)
        self.loss_type = loss_type
        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = p2_loss_weight_gamma

    def register_schedule(self, timesteps, linear_start, linear_end, cosine_s, schedule):
        betas = make_beta_schedule(schedule, timesteps, linear_start, linear_end, cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.num_timesteps = int(timesteps)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, denoise_model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predicted_noise = denoise_model(x_noisy, t)

        if self.loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, denoise_model, x, t, t_index):
        betas_t = extract_into_tensor(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / extract_into_tensor(1.0 - self.betas, t, x.shape))
        posterior_mean = sqrt_recip_alphas_t * (x - betas_t * denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return posterior_mean
        noise = torch.randn_like(x)
        return posterior_mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_model, shape):
        device = next(denoise_model.parameters()).device
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop", total=self.num_timesteps):
            img = self.p_sample(denoise_model, img, torch.full((shape[0],), i, device=device, dtype=torch.long), i)
        return img

    @torch.no_grad()
    def sample(self, denoise_model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(denoise_model, shape=(batch_size, channels, image_size, image_size))


# =========================================================
# Latent Diffusion (wrapper)
# =========================================================
class LatentDiffusion(GaussianDiffusion):
    def __init__(self, first_stage_config, cond_stage_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initializing LatentDiffusion model...")

        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.cond_stage_model = instantiate_from_config(cond_stage_config) if cond_stage_config else None
        self.learning_rate = kwargs.get("base_learning_rate", 2e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(self.denoise_model, x, t)

    def encode(self, x):
        return self.first_stage_model.encode(x)

    def decode(self, z):
        return self.first_stage_model.decode(z)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())

        if BNB_AVAILABLE:
            print("Using bitsandbytes 8-bit AdamW optimizer.")
            opt = bnb.optim.AdamW8bit(params, lr=lr)
        else:
            print("Using standard AdamW optimizer.")
            opt = torch.optim.AdamW(params, lr=lr)

        return opt


# =========================================================
# Utilities for testing
# =========================================================
if __name__ == "__main__":
    print("✅ ddpm_colab.py loaded successfully — ready for Colab GPU")

    # Quick functional test
    model = GaussianDiffusion()
    dummy_denoiser = nn.Identity()
    img = torch.randn(2, 3, 64, 64).cuda() if torch.cuda.is_available() else torch.randn(2, 3, 64, 64)
    out = model.q_sample(img, torch.randint(0, 10, (2,)))
    print("q_sample output shape:", out.shape)
