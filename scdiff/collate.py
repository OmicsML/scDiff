import collections
import contextlib
import re
import torch

from typing import Callable, Dict, Optional, Tuple, Type, Union

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
            keys_to_keep: list = None):
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        keys_to_keep = keys_to_keep if keys_to_keep is not None else list(elem)
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in keys_to_keep})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in keys_to_keep}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.as_tensor(batch)


def collate_float_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch, dtype=torch.float64)


def collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch)


def collate_str_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return batch


default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}
with contextlib.suppress(ImportError):
    import numpy as np
    # For both ndarray and memmap (subclass of ndarray)
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn

class CollateDictionary:
    """Collate dictionay with keys to ignore"""
    def __init__(self, keys_to_ignore=None):
        self.keys_to_ignore = keys_to_ignore

    def __call__(self, batch):
        keys = list(batch[0])
        if self.keys_to_ignore is not None:
            keys_to_ignore = [x for x in self.keys_to_ignore if x in keys]
            keys_to_keep = [x for x in keys if x not in keys_to_ignore]
            batch_collated = collate(batch, collate_fn_map=default_collate_fn_map, keys_to_keep=keys_to_keep)
            for k in keys_to_ignore:
                batch_collated[k] = batch[0][k]
        else:
            batch_collated = collate(batch, collate_fn_map=default_collate_fn_map)
        return batch_collated
