import numpy as np
import torch
from math import pi
from torch import nn

# Interpolation kernels
def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))

def lanczos2(x):
    x_safe = np.where(x==0, np.finfo(np.float32).eps, x)
    return ((np.sin(pi * x_safe) * np.sin(pi * x_safe / 2)) / ((pi**2 * x_safe**2 / 2))) * (np.abs(x) < 2)

def lanczos3(x):
    x_safe = np.where(x==0, np.finfo(np.float32).eps, x)
    return ((np.sin(pi * x_safe) * np.sin(pi * x_safe / 3)) / ((pi**2 * x_safe**2 / 3))) * (np.abs(x) < 3)

def box(x):
    return ((-0.5 <= x) & (x < 0.5)).astype(np.float32)

def linear(x):
    return ((x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))).astype(np.float32)


class Resizer(nn.Module):
    def __init__(self, in_shape, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
        super().__init__()

        # Determine final scale and output size
        scale_factor, output_shape = self.fix_scale_and_size(in_shape, output_shape, scale_factor)

        # Select kernel function
        method, kernel_width = {
            "cubic": (cubic, 4.0),
            "lanczos2": (lanczos2, 4.0),
            "lanczos3": (lanczos3, 6.0),
            "box": (box, 1.0),
            "linear": (linear, 2.0),
            None: (cubic, 4.0)
        }.get(kernel)

        # Only antialias when downscaling
        antialiasing *= np.any(np.array(scale_factor) < 1)

        # Sort dims by scale factor for efficiency
        sorted_dims = np.argsort(np.array(scale_factor))
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        # Precompute weights and field of view for each dim
        self.weights = nn.ParameterList()
        self.field_of_view = nn.ParameterList()

        for dim in self.sorted_dims:
            weights, fov = self.contributions(
                in_shape[dim], output_shape[dim], scale_factor[dim], method, kernel_width, antialiasing
            )

            weights = torch.tensor(weights.T, dtype=torch.float32)
            weights = nn.Parameter(weights.reshape(*weights.shape, *(len(scale_factor)-1)*[1]), requires_grad=False)
            fov = nn.Parameter(torch.tensor(fov.T.astype(np.int32), dtype=torch.long), requires_grad=False)

            self.weights.append(weights)
            self.field_of_view.append(fov)

    def forward(self, x):
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
            x = torch.transpose(x, dim, 0)
            x = torch.sum(x[fov] * w, dim=0)
            x = torch.transpose(x, dim, 0)
        return x

    def fix_scale_and_size(self, in_shape, output_shape, scale_factor):
        if scale_factor is not None:
            if np.isscalar(scale_factor) and len(in_shape) > 1:
                scale_factor = [scale_factor] * 2
            scale_factor = [1] * (len(in_shape) - len(scale_factor)) + list(scale_factor)

        if output_shape is not None:
            output_shape = list(in_shape[len(output_shape):]) + list(np.uint(np.array(output_shape)))

        if scale_factor is None:
            scale_factor = 1.0 * np.array(output_shape) / np.array(in_shape)

        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(in_shape) * np.array(scale_factor)))

        return scale_factor, output_shape

    def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        kernel_width = kernel_width / scale if antialiasing else kernel_width

        out_coords = np.arange(1, out_length + 1)
        shifted_out_coords = out_coords - (out_length - in_length * scale) / 2
        match_coords = shifted_out_coords / scale + 0.5 * (1 - 1/scale)
        left_boundary = np.floor(match_coords - kernel_width / 2)
        expanded_width = np.ceil(kernel_width) + 2
        field_of_view = np.squeeze(np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_width) - 1))

        weights = fixed_kernel(np.expand_dims(match_coords, axis=1) - field_of_view - 1)
        weights /= np.sum(weights, axis=1, keepdims=True).clip(min=1.0)

        mirror = np.uint32(np.concatenate([np.arange(in_length), np.arange(in_length-1, -1, -1)]))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

        return weights, field_of_view
