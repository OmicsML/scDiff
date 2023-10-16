# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from einops import repeat

from scdiff.utils.modules import sinusoidal_embedding


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        embedding = sinusoidal_embedding(timesteps, dim, max_period)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class MaskedEncoderConditioner(nn.Module):
    """Use 2-layer MLP to encoder available feature number.

    The encoded feature number condition is added to the cell embddings. If
    disabled, then directly return the original cell embeddings.

    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        use_ratio: bool = False,
        use_se: bool = False,
        use_semlp: bool = False,
        concat: bool = False,
        disable: bool = False,
    ):
        super().__init__()
        assert not (use_ratio and use_se), "Cannot set use_se and use_ratio together"
        assert not (use_se and use_semlp), "Cannot set use_se and use_semlp together"
        assert not (use_se and concat), "Cannot set use_se and concat together"
        self.dim = dim
        self.use_ratio = use_ratio
        self.use_se = use_se or use_semlp
        self.concat = concat
        self.disable = disable
        if not disable:
            dim_in = dim if self.use_se else 1
            dim_in = dim_in + dim if concat else dim_in
            dim_hid = dim * mult

            self.proj = nn.Sequential(
                nn.Linear(dim_in, dim_hid),
                nn.SiLU(),
                nn.Linear(dim_hid, dim),
            ) if not use_se else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.disable and mask is not None:
            # Count the number of denoising input featues
            size = (mask.bool()).sum(1, keepdim=True).float()

            if self.use_ratio:
                h = size / x.shape[1]
            elif self.use_se:
                h = sinusoidal_embedding(size.ravel(), dim=self.dim, max_period=x.shape[1] + 1)
            else:
                h = size

            if self.concat:
                h = torch.cat((x, h), dim=-1)
                x = self.proj(h)
            else:
                h = self.proj(h)
                x = x + h

        return x


class ConditionEncoderWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        x = torch.cat((x, context), dim=1).sum(1) if context is not None else x.squeeze(1)
        return self.module(x).unsqueeze(1)
