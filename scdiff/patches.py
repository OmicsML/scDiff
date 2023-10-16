"""A collection of patches to temporarily bridge compatibilities.

Parts pulled from:
1. https://github.com/Lightning-AI/lightning/blob/5340d960b9427dcfe5fb953cac0e86763d423546/src/lightning/pytorch/utilities/argparse.py

"""
__all__ = [
    "pl_trainer_add_argparse_args",
    "rank_zero_only",
]

import inspect
import warnings
from argparse import ArgumentParser, Namespace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
)

import pytorch_lightning as pl
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except ModuleNotFoundError:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only


def pl_trainer_add_argparse_args(parser: ArgumentParser) -> ArgumentParser:
    warnings.warn(
        "Please replace add_argparse_args with a better solution soon",
        DeprecationWarning,
        stacklevel=2,
    )

    if hasattr(pl.Trainer, "add_argparse_args"):
        return pl.Trainer.add_argparse_args(parser)

    def _int_or_float_type(x: Union[int, float, str]) -> Union[int, float]:
        if "." in str(x):
            return float(x)
        return int(x)

    def str_to_bool_or_str(val: str) -> Union[str, bool]:
        lower = val.lower()
        if lower in ("y", "yes", "t", "true", "on", "1"):
            return True
        if lower in ("n", "no", "f", "false", "off", "0"):
            return False
        return val

    def str_to_bool(val: str) -> bool:
        val_converted = str_to_bool_or_str(val)
        if isinstance(val_converted, bool):
            return val_converted
        raise ValueError(f"invalid truth value {val_converted}")

    def str_to_bool_or_int(val: str) -> Union[bool, int, str]:
        val_converted = str_to_bool_or_str(val)
        if isinstance(val_converted, bool):
            return val_converted
        try:
            return int(val_converted)
        except ValueError:
            return val_converted

    def _precision_allowed_type(x: Union[int, str]) -> Union[int, str]:
        try:
            return int(x)
        except ValueError:
            return x

    def get_init_arguments_and_types(
        cls,
        ignore_arg_names: List[str],
    ) -> List[Tuple[str, Tuple, Any]]:
        cls_default_params = inspect.signature(cls).parameters
        name_type_default = []
        for arg in cls_default_params:
            if arg in ignore_arg_names:
                continue

            arg_type = cls_default_params[arg].annotation
            arg_default = cls_default_params[arg].default
            try:
                if type(arg_type).__name__ == "_LiteralGenericAlias":
                    # Special case: Literal[a, b, c, ...]
                    arg_types = tuple({type(a) for a in arg_type.__args__})
                elif "typing.Literal" in str(arg_type) or "typing_extensions.Literal" in str(arg_type):
                    # Special case: Union[Literal, ...]
                    arg_types = tuple({type(a) for union_args in arg_type.__args__ for a in union_args.__args__})
                else:
                    # Special case: ComposedType[type0, type1, ...]
                    arg_types = tuple(arg_type.__args__)
            except (AttributeError, TypeError):
                arg_types = (arg_type,)

            name_type_default.append((arg, arg_types, arg_default))

        return name_type_default

    ignore_arg_names = ["self", "args", "kwargs"]
    allowed_types = (str, int, float, bool)

    # Get symbols from cls or init function.
    for symbol in (pl.Trainer, pl.Trainer.__init__):
        args_and_types = get_init_arguments_and_types(symbol, ignore_arg_names)  # type: ignore[arg-type]
        if len(args_and_types) > 0:
            break

    for arg, arg_types, arg_default in args_and_types:
        arg_types = tuple(at for at in allowed_types if at in arg_types)
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs: Dict[str, Any] = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type: Callable[[str], Union[bool, int, float, str]] = str_to_bool
            elif int in arg_types:
                use_type = str_to_bool_or_int
            elif str in arg_types:
                use_type = str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == "devices":
            use_type = str

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == "track_grad_norm":
            use_type = float

        # hack for precision
        if arg == "precision":
            use_type = _precision_allowed_type

        parser.add_argument(
            f"--{arg}",
            dest=arg,
            default=arg_default,
            type=use_type,
            required=(arg_default == inspect._empty),
            **arg_kwargs,
        )

    return parser


def pl_trainer_from_argparse_args(args: Namespace, **kwargs: Any) -> pl.Trainer:
    warnings.warn(
        "Please replace from_argparse_args with a better solution soon",
        DeprecationWarning,
        stacklevel=2,
    )

    if hasattr(pl.Trainer, "from_argparse_args"):
        return pl.Trainer.from_argparse_args(args, **kwargs)

    if isinstance(args, ArgumentParser):
        raise NotImplementedError

    params = vars(args)

    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
    trainer_kwargs.update(**kwargs)

    return pl.Trainer(**trainer_kwargs)
