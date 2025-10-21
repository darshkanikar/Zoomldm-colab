import torch
import numpy as np

# -----------------------------
# BASE DISTRIBUTIONS
# -----------------------------
class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(AbstractDistribution):
    """
    Diagonal Gaussian with optional deterministic mode.
    parameters: tensor of shape [B, 2*C, H, W] (mean + logvar concatenated)
    """
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            device = self.parameters.device
            self.std = torch.zeros_like(self.mean, device=device)
            self.var = torch.zeros_like(self.mean, device=device)

    def sample(self):
        if self.deterministic:
            return self.mean
        device = self.parameters.device
        return self.mean + self.std * torch.randn_like(self.mean, device=device)

    def kl(self, other=None):
        if self.deterministic:
            return torch.zeros(self.mean.size(0), device=self.mean.device)
        if other is None:
            # KL against standard normal
            return 0.5 * torch.sum(self.mean**2 + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                (self.mean - other.mean)**2 / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3]
            )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.zeros(sample.size(0), device=sample.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + (sample - self.mean)**2 / self.var,
            dim=dims
        )

    def mode(self):
        return self.mean


# -----------------------------
# UTILITY FUNCTION
# -----------------------------
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two diagonal Gaussians:
    KL(N(mean1, var1) || N(mean2, var2))
    Broadcasting supported.
    """
    # Ensure tensors
    tensor = next(obj for obj in (mean1, logvar1, mean2, logvar2) if isinstance(obj, torch.Tensor))
    logvar1 = logvar1 if isinstance(logvar1, torch.Tensor) else torch.tensor(logvar1, device=tensor.device)
    logvar2 = logvar2 if isinstance(logvar2, torch.Tensor) else torch.tensor(logvar2, device=tensor.device)

    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
