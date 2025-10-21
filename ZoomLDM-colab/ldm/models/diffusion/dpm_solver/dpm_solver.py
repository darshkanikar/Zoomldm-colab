import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

# ----------------------
# Noise Schedule VP
# ----------------------
class NoiseScheduleVP:
    def __init__(self, schedule='discrete', betas=None, alphas_cumprod=None,
                 continuous_beta_0=0.1, continuous_beta_1=20.):
        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(f"Unsupported noise schedule {schedule}")

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1))
            self.log_alpha_array = log_alphas.reshape((1, -1))
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            if schedule == 'cosine':
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            return log_alpha_fn(t) - self.cosine_log_alpha_0

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            return t_fn(log_alpha)

# ----------------------
# Helper functions
# ----------------------
def interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        x_idx == 0,
        torch.tensor(1, device=x.device),
        torch.where(x_idx == K, torch.tensor(K - 2, device=x.device), cand_start_idx)
    )
    end_idx = torch.where(start_idx == cand_start_idx, start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, 2, start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, 2, end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        x_idx == 0,
        torch.tensor(0, device=x.device),
        torch.where(x_idx == K, torch.tensor(K - 2, device=x.device), cand_start_idx)
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, 2, start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, 2, (start_idx2 + 1).unsqueeze(2)).squeeze(2)
    return start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)

def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]
