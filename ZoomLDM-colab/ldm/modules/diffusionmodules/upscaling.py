import torch
import torch.nn as nn
import numpy as np
from functools import partial

# -----------------------------
# UTILITY FUNCTIONS (Colab-ready)
# -----------------------------
def default(val, d):
    return val if val is not None else d()

def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy/tensor array for given timesteps
    and reshape to broadcast with `broadcast_shape`.
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr).float()
    out = arr[timesteps].to(dtype=torch.float32)
    while len(out.shape) < len(broadcast_shape):
        out = out[..., None]
    return out.expand(broadcast_shape)

def make_beta_schedule(schedule, timesteps, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        return np.linspace(linear_start, linear_end, timesteps, dtype=np.float32)
    elif schedule == "cosine":
        t = np.linspace(0, timesteps, timesteps + 1, dtype=np.float64)
        alphas_cumprod = np.cos(((t / timesteps) + cosine_s) / (1 + cosine_s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule}")

# -----------------------------
# BASE MODEL
# -----------------------------
class AbstractLowScaleModel(nn.Module):
    """For concatenating a downsampled image to the latent representation"""
    def __init__(self, noise_schedule_config=None):
        super().__init__()
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(
        self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        betas = make_beta_schedule(
            beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])
        self.linear_start = linear_start
        self.linear_end = linear_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, x):
        return x, None

    def decode(self, x):
        return x

# -----------------------------
# SIMPLE VARIANTS
# -----------------------------
class SimpleImageConcat(AbstractLowScaleModel):
    """No noise level conditioning"""
    def __init__(self):
        super().__init__(noise_schedule_config=None)
        self.max_noise_level = 0

    def forward(self, x):
        return x, torch.zeros(x.shape[0], device=x.device).long()


class ImageConcatWithNoiseAugmentation(AbstractLowScaleModel):
    def __init__(self, noise_schedule_config=None, max_noise_level=1000):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def forward(self, x, noise_level=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        z = self.q_sample(x, noise_level)
        return z, noise_level

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    model = ImageConcatWithNoiseAugmentation(noise_schedule_config={"beta_schedule": "linear", "timesteps": 1000})
    x = torch.randn(2, 3, 64, 64)
    z, levels = model(x)
    print("Output shape:", z.shape)
    print("Noise levels:", levels)
