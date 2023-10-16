import math
from inspect import getfullargspec
from typing import Any, List, Optional

import torch
import torch.nn as nn


def create_activation(name):
    if name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name, n, h=16):
    if name is None:
        return nn.Identity()
    elif name == "layernorm":
        return nn.LayerNorm(n)
    elif name == "batchnorm":
        return nn.BatchNorm1d(n)
    elif name == "groupnorm":
        return nn.GroupNorm(h, n)
    elif name.startswith("groupnorm"):
        inferred_num_groups = int(name.repalce("groupnorm", ""))
        return nn.GroupNorm(inferred_num_groups, n)
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def sinusoidal_embedding(pos: torch.Tensor, dim: int, max_period: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=pos.device)
    args = pos[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    if repeat:
        noise = torch.randn((1, *shape[1:]), device=device)
        repeat_noise = noise.repeat(shape[0], *((1,) * (len(shape) - 1)))
        return repeat_noise
    else:
        return torch.randn(shape, device=device)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class BatchedOperation:
    """Wrapper to expand batched dimension for input tensors.

    Args:
        batch_dim: Which dimension the batch goes.
        plain_num_dim: Number of dimensions for plain (i.e., no batch) inputs,
            which is used to determine whether the input the batched or not.
        ignored_args: Which arguments to ignored for automatic batch dimension
            expansion.
        squeeze_output_batch: If set to True, then try to squeeze out the batch
            dimension of the output tensor.

    """

    def __init__(
        self,
        batch_dim: int = 0,
        plain_num_dim: int = 2,
        ignored_args: Optional[List[str]] = None,
        squeeze_output_batch: bool = True,
    ):
        self.batch_dim = batch_dim
        self.plain_num_dim = plain_num_dim
        self.ignored_args = set(ignored_args or [])
        self.squeeze_output_batch = squeeze_output_batch
        self._is_batched = None

    def __call__(self, func):
        arg_names = getfullargspec(func).args

        def bounded_func(*args, **kwargs):
            new_args = []
            for arg_name, arg in zip(arg_names, args):
                if self.unsqueeze_batch_dim(arg_name, arg):
                    arg = arg.unsqueeze(self.batch_dim)
                new_args.append(arg)

            for arg_name, arg in kwargs.items():
                if self.unsqueeze_batch_dim(arg_name, arg):
                    kwargs[arg_name] = arg.unsqueeze(self.batch_dim)

            out = func(*new_args, **kwargs)

            if self.squeeze_output_batch:
                out = out.squeeze(self.batch_dim)

            return out

        return bounded_func

    def unsqueeze_batch_dim(self, arg_name: str, arg_val: Any) -> bool:
        return (
            isinstance(arg_val, torch.Tensor)
            and (arg_name not in self.ignored_args)
            and (not self.is_batched(arg_val))
        )

    def is_batched(self, val: torch.Tensor) -> bool:
        num_dim = len(val.shape)
        if num_dim == self.plain_num_dim:
            return False
        elif num_dim == self.plain_num_dim + 1:
            return True
        else:
            raise ValueError(
                f"Tensor should have either {self.plain_num_dim} or "
                f"{self.plain_num_dim + 1} number of dimension, got {num_dim}",
            )


# FIX: depreacte this, replace with BatchedOperation
def batch_apply_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 2:  # (length, channel)
        return norm(x)
    elif len(x.shape) == 3:  # (batch, length, channel)
        if isinstance(norm, nn.Identity):
            return x
        elif isinstance(norm, nn.LayerNorm):
            return norm(x)
        elif isinstance(norm, (nn.BatchNorm1d, nn.GroupNorm)):
            return norm(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            raise NotImplementedError(f"{norm!r} not supported yet")
    else:
        raise ValueError(f"Invalid dimension of x: {x.shape=}")
