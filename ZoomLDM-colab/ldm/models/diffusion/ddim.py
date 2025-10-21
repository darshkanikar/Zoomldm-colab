"""
DDIM Sampler (Colab Compatible)
-------------------------------
Patched for:
- PyTorch ≥ 2.1 (Colab default)
- Python 3.12+
- Single GPU or CPU
- Mixed precision inference
- Device-safe buffer registration
"""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)


class DDIMSampler:
    def __init__(self, model, schedule="linear", device=None):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------
    # Utility to safely register buffers
    # -----------------------------------
    def register_buffer(self, name, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        setattr(self, name, tensor.to(self.device, dtype=torch.float32))

    # -----------------------------------
    # Build DDIM schedule
    # -----------------------------------
    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0):
        # Create timesteps
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
        )

        alphas_cumprod = self.model.alphas_cumprod
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)

        # Precompute constants
        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1.0)))

        # DDIM parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta
        )

        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", torch.sqrt(1.0 - ddim_alphas))

        sigmas_orig = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_orig)

    # -----------------------------------
    # Sampling entry point
    # -----------------------------------
    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        eta=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        callback=None,
        img_callback=None,
        quantize_x0=False,
        temperature=1.0,
    ):
        """Perform full DDIM sampling loop"""
        self.make_schedule(S, ddim_eta=eta)

        C, H, W = shape
        x_T = torch.randn((batch_size, C, H, W), device=self.device)

        samples, intermediates = self.ddim_sampling(
            conditioning,
            (batch_size, C, H, W),
            x_T=x_T,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            temperature=temperature,
        )
        return samples, intermediates

    # -----------------------------------
    # Core sampling loop
    # -----------------------------------
    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T,
        quantize_denoised=False,
        callback=None,
        img_callback=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        temperature=1.0,
    ):
        b = shape[0]
        img = x_T.clone()
        intermediates = {"x_inter": [img], "pred_x0": [img]}

        iterator = tqdm(
            reversed(self.ddim_timesteps),
            total=len(self.ddim_timesteps),
            desc="DDIM Sampling",
        )

        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            img, pred_x0 = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=i,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
            )

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            if i % max(1, len(self.ddim_timesteps) // 10) == 0 or i == len(self.ddim_timesteps) - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    # -----------------------------------
    # One denoising step
    # -----------------------------------
    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        quantize_denoised=False,
        temperature=1.0,
    ):
        b, *_, device = *x.shape, x.device

        # -------------
        # Model output
        # -------------
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                model_output = self.model.apply_model(x, t, c)
            else:
                # Classifier-free guidance (unconditional + conditional)
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(c, dict):
                    c_in = {k: torch.cat([unconditional_conditioning[k], c[k]]) for k in c}
                elif isinstance(c, list):
                    c_in = [torch.cat([u, v]) for u, v in zip(unconditional_conditioning, c)]
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        # -------------
        # DDIM update
        # -------------
        e_t = (
            model_output
            if self.model.parameterization != "v"
            else self.model.predict_eps_from_z_and_v(x, t, model_output)
        )

        # Retrieve parameters
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=device
        )

        # Predict x0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, *_ = self.model.first_stage_model.quantize(pred_x0)

        # Compute new sample
        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma_t**2, min=0.0)) * e_t
        noise = sigma_t * noise_like(x.shape, device) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0


# ------------------------------------------------------------
# ✅ Quick check (safe import)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("✅ DDIMSampler (Colab version) loaded successfully.")
