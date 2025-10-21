import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F

# Postprocess a generated large image using diffusion at highest magnification (20x)
MAG_DICT = {
    "20x": 0,
    "10x": 1,
    "5x": 2,
    "2_5x": 3,
    "1_25x": 4,
    "0_625x": 5,
    "0_3125": 6,
    "0.15625": 7,
}

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Spherical linear interpolation between v0 and v1"""
    inputs_are_torch = False
    if isinstance(v0, torch.Tensor):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)
    return v2

def get_conditioning(model, i, j, embeddings, uncond=False, patch_size=64, embedding_spatial=(16, 16)):
    """Compute interpolated conditioning embedding for a patch"""
    if uncond:
        return torch.zeros((1, embeddings.shape[0], 1, 1), device=model.device)

    # Pad embeddings
    emb_h, emb_w = embedding_spatial
    embeddings_padded = F.pad(
        embeddings.reshape(1, -1, emb_h, emb_w),
        (0, 1, 0, 1), mode="replicate"
    ).view(-1, emb_h + 1, emb_w + 1)

    i1, i2 = (i // patch_size,) * 2
    i3, i4 = (i // patch_size + 1,) * 2
    j1, j3 = (j // patch_size,) * 2
    j2, j4 = (j // patch_size + 1,) * 2

    e1 = embeddings_padded[:, i1, j1]
    e2 = embeddings_padded[:, i2, j2]
    e3 = embeddings_padded[:, i3, j3]
    e4 = embeddings_padded[:, i4, j4]

    t1 = (j / patch_size - j1) / max(1, j2 - j1)
    t2 = (i / patch_size - i1) / max(1, i3 - i1)

    e_top = slerp(t1, e1, e2)
    e_bot = slerp(t1, e3, e4)
    e_interp = slerp(t2, e_top, e_bot).view(-1, 1, 1)

    # Normalize embedding
    e_interp = (e_interp - e_interp.mean(0, keepdim=True)) / e_interp.std(0, keepdim=True)
    cond_dict_20x = dict(
        ssl_feat=[e_interp],
        mag=torch.tensor([MAG_DICT["20x"]], device=model.device).long()
    )
    return model.get_learned_conditioning(cond_dict_20x)

def postprocess_image(model, xt_20x_all, ssl_feat, t0, stride=50, guidance=3.0, sliding_window_size=16, emb_h=4, emb_w=4, batch_size=16):
    device = model.device
    # Add initial noise
    atbar = model.alphas_cumprod[t0-1].view(1,1,1,1).to(device)
    xt_20x_all_postprocessed = torch.sqrt(atbar)*xt_20x_all.clone() + torch.sqrt(1-atbar)*torch.randn_like(xt_20x_all)
    x = rearrange(xt_20x_all_postprocessed, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=emb_h, p2=emb_w)

    lt_sz = 64
    img_cond_list = []
    no_cond_list = []

    # Compute conditioning for sliding windows
    for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
        for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
            with torch.no_grad():
                img_cond_list.append(get_conditioning(model, j, k, ssl_feat, uncond=False, embedding_spatial=(emb_h, emb_w)))
                no_cond_list.append(get_conditioning(model, j, k, ssl_feat, uncond=True, embedding_spatial=(emb_h, emb_w)))

    img_cond_list = torch.vstack(img_cond_list)
    no_cond_list = torch.vstack(no_cond_list)
    batch_size = len(img_cond_list)

    # Diffusion postprocessing loop
    for t in range(t0, 0, -stride):
        atbar = model.alphas_cumprod[t-1].view(1,1,1,1).to(device)
        atbar_prev = model.alphas_cumprod[max(t-1-stride,0)].view(1,1,1,1).to(device)
        beta_tilde = (model.betas[t-1] * (1 - atbar_prev) / (1 - atbar)).view(1,1,1,1).to(device)

        x = rearrange(xt_20x_all_postprocessed, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=emb_h, p2=emb_w)

        # Sliding-window denoising
        eps_map = torch.zeros_like(x)
        x0_map = torch.zeros_like(x)
        avg_map = torch.zeros_like(x)

        x_slice_list = []
        indices_map = {}
        idx_counter = 0
        for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
            for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
                x_slice_list.append(x[:, :, j:j+lt_sz, k:k+lt_sz])
                indices_map[j, k] = idx_counter
                idx_counter += 1

        x_slice_list = torch.vstack(x_slice_list)

        # Process slices in batches
        cond_out_all = []
        uncond_out_all = []
        for i in range(0, x_slice_list.shape[0], batch_size):
            x_batch = x_slice_list[i:i+batch_size]
            cond_batch = torch.vstack([img_cond_list[i:i+batch_size], no_cond_list[i:i+batch_size]])
            x_batch_combined = torch.vstack([x_batch]*2)
            t_cond = torch.tensor([t]*x_batch_combined.shape[0], device=device)

            with torch.cuda.amp.autocast():
                out = model.model.diffusion_model(x_batch_combined, t_cond.float(), context=cond_batch)
                cond_out, uncond_out = torch.tensor_split(out, 2)
                cond_out_all.append(cond_out)
                uncond_out_all.append(uncond_out)

        cond_out_all = torch.cat(cond_out_all, dim=0)
        uncond_out_all = torch.cat(uncond_out_all, dim=0)

        epsilon_combined = (1 + guidance) * cond_out_all - guidance * uncond_out_all
        x0_combined = (x_slice_list / torch.sqrt(atbar)) - epsilon_combined * torch.sqrt((1 - atbar)/atbar)

        # Reconstruct full image from slices
        for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
            for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
                idx = indices_map[j, k]
                eps_map[:, :, j:j+lt_sz, k:k+lt_sz] += epsilon_combined[idx]
                x0_map[:, :, j:j+lt_sz, k:k+lt_sz] += x0_combined[idx]
                avg_map[:, :, j:j+lt_sz, k:k+lt_sz] += 1

        x0_pred = x0_map / avg_map
        epsilon = (x - torch.sqrt(atbar) * x0_pred) / torch.sqrt(1 - atbar)

        # Predict next step
        x_prev = torch.sqrt(atbar_prev) * x0_pred + torch.sqrt(1-atbar_prev-beta_tilde) * epsilon + torch.sqrt(beta_tilde) * torch.randn_like(x)
        xt_20x_all_postprocessed = rearrange(x_prev, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', p1=emb_h, p2=emb_w)

    return xt_20x_all_postprocessed
