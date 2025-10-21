import torch
from einops import rearrange

# ---------------------------
# Model prediction wrapper
# ---------------------------
def model_pred(model, xt, t, cond, w=0):
    t_cond = torch.full((xt.shape[0],), float(t), device=model.device)
    with torch.cuda.amp.autocast():
        with model.ema_scope():
            with torch.no_grad():
                if w != 0:
                    bs = xt.shape[0]
                    pred_eps = model.model.diffusion_model(
                        torch.cat([xt, xt], dim=0),
                        torch.cat([t_cond, t_cond], dim=0),
                        torch.cat([cond, torch.zeros_like(cond)], dim=0)
                    )
                    pred_eps = (w + 1) * pred_eps[:bs] - w * pred_eps[bs:]
                else:
                    pred_eps = model.model.diffusion_model(xt, t_cond, cond)
    return pred_eps


# ---------------------------
# Gaussian blending kernel
# ---------------------------
def gaussian_kernel(size=64, mu=0, sigma=1):
    coords = torch.linspace(-1, 1, size)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0)  # shape: [2, size, size]
    d = torch.linalg.norm(grid - mu, dim=0)
    kernel = torch.exp(-(d**2) / sigma**2)
    kernel /= kernel.max()
    return kernel.float()


# ---------------------------
# Decode large latent image with sliding window
# ---------------------------
def decode_large_image(latent, model, sliding_window_size=16, sigma=0.8):
    f = 4  # upsampling factor of first stage decoder
    lt_sz = 64
    device = latent.device
    out_H, out_W = f * latent.shape[2], f * latent.shape[3]
    out_img = torch.zeros((latent.shape[0], 3, out_H, out_W), device=device)
    avg_map = torch.zeros_like(out_img)

    kernel = gaussian_kernel(size=f * lt_sz, sigma=sigma).to(device)
    kernel = kernel.view(1, 1, f * lt_sz, f * lt_sz)

    for i in range(0, latent.shape[2] - lt_sz + 1, sliding_window_size):
        for j in range(0, latent.shape[3] - lt_sz + 1, sliding_window_size):
            with torch.no_grad():
                patch = latent[:, :, i:i+lt_sz, j:j+lt_sz]
                decoded = model.decode_first_stage(patch)
                out_img[:, :, i*f:(i+lt_sz)*f, j*f:(j+lt_sz)*f] += decoded * kernel
                avg_map[:, :, i*f:(i+lt_sz)*f, j*f:(j+lt_sz)*f] += kernel

    out_img /= avg_map
    out_img = torch.clamp((out_img + 1) / 2.0, 0.0, 1.0)
    out_img = (out_img * 255).to(torch.uint8)
    return out_img.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
