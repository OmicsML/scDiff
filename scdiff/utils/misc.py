import os
import importlib
import warnings
from inspect import isfunction
from pprint import pformat
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
)

import numpy as np
import torch
from omegaconf import DictConfig

from scdiff.typing import TensorArray


def as_tensor(x: TensorArray, assert_type: bool = False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor) and assert_type:
        raise TypeError(f"Expecting tensor or numpy array, got, {type(x)}")
    return x


def as_1d_vec(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)
    elif len(x.shape) == 1:
        raise ValueError(f"input must be one or two dimensional tensor, got {x.shape}")
    return x


def as_array(x: TensorArray, assert_type: bool = False):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray) and assert_type:
        raise TypeError(f"Expecting tensor or numpy array, got, {type(x)}")
    return x


def check_str_option(
    input_name: str,
    input_opt: Optional[str],
    available_opts: Union[List[str], Tuple[str], Set[str], Any],
    optional: bool = True,
    warn_fail: bool = True,
) -> str:
    """Pass through an input option and raise ValueError if it is in invalid."""
    if not isinstance(available_opts, (list, tuple, set)):
        try:
            available_opts = get_args(available_opts)
        except Exception:
            if warn_fail:
                warnings.warn(
                    f"Fail to check option for {input_name}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if input_opt is None:
        if not optional:
            raise ValueError(f"{input_name} can not be None.")
    elif input_opt not in available_opts:
        raise ValueError(
            f"Unknown option {input_opt!r} for {input_name}. "
            f"Available options are: {available_opts}.",
        )

    return input_opt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def list_exclude(input_list: List[Any], exclude_list: Optional[Union[List[Any], Set[Any]]]) -> List[Any]:
    if exclude_list is not None:
        exclude_set = set(exclude_list)
        input_list = [i for i in input_list if i not in exclude_set]
    return input_list


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(
    config: Union[Dict, DictConfig, str],
    _target_key: str = "target",
    _params_key: str = "params",
    _catch_conflict: bool = True,
    **extra_kwargs: Any,
):
    # Check target specificiation and handel special conditions
    if _target_key not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError(f"Expected key `{_target_key}` to instantiate.")

    # Obtain target object and kwargs
    cls = get_obj_from_str(config["target"])
    kwargs = config.get(_params_key, dict())

    # Check conflict and merge kwargs
    if (common_keys := sorted(set(kwargs) & set(extra_kwargs))):
        diff_keys = []
        for key in common_keys:
            if kwargs[key] != extra_kwargs[key]:
                diff_keys.append(key)

        if diff_keys and _catch_conflict:
            conflicting_config_kwargs = {i: kwargs[i] for i in diff_keys}
            conflicting_extra_kwargs = {i: extra_kwargs[i] for i in diff_keys}
            raise ValueError(
                "Conflicting parameters between configs and those that are "
                "additionally specified. Please resolve or set _catch_conflict "
                f"to False to bypass this issue.\n{conflicting_config_kwargs=}\n"
                f"{conflicting_extra_kwargs=}\n",
            )
    kwargs = {**kwargs, **extra_kwargs}

    # Instantiate object and handel exception during instantiation
    try:
        return cls(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate {cls!r} with kwargs:\n{pformat(kwargs)}") from e


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
