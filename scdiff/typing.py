__all__ = [
    "TensorArray",
]

from typing import Union

from numpy import ndarray as Array
from torch import Tensor


TensorArray = Union[Tensor, Array]
