# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2022 HeliXon Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""

"""
# =============================================================================
# Imports
# =============================================================================
import numbers
import typing
from typing import *

import torch
from torch.nn import functional as F
import numpy as np
# =============================================================================
# Constants
# =============================================================================

T = typing.TypeVar("T")


# =============================================================================
# Functions
# =============================================================================
def mask2bias(mask: torch.Tensor, *, inf: float = 1e9) -> torch.Tensor:
    """Convert mask to attention bias

    Args:
        mask:
        inf:

    Returns:

    """
    return mask.sub(1).mul(inf)


def normalize(
        inputs: torch.Tensor,
        normalized_shape: typing.Optional[
            typing.Union[int, typing.List[int], torch.Size]] = None
) -> torch.Tensor:
    """Layer normalization without a module (and weight)

    This is mostly used for layer normalization that is placed directly before
    a linear operation

    Args:
        inputs: the input tensor to be normalized
        normalized_shape: the normalized_shape for normalization

    Returns:

    """
    if normalized_shape is None:
        normalized_shape = inputs.shape[-1]
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)

    return F.layer_norm(inputs, normalized_shape, None, None, 1e-5)


def masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: typing.Union[int, typing.Sequence[int], None],
        keepdim: typing.Optional[bool] = False,
        eps: typing.Optional[float] = 4e-5
) -> torch.Tensor:
    """

    Args:
        values: the values to take the mean for
        mask: the mask to take the mean with
        dim: the dimension along which to take the mean
        keepdim: to keep the dimension
        eps: the epsilon to compute mean for

    Returns:
        mean result

    """
    values = values.masked_fill(~mask.bool(), 0).sum(dim, keepdim=keepdim)
    norm = mask.sum(dim, keepdim=keepdim, dtype=values.dtype) + eps
    return values / norm


def recursive_to(obj: typing.Any, **kwargs) -> typing.Any:
    r"""
    Just to move things to space
    *args is removed because it brings problems in using .cpu()

    Args:
        obj (): the object to move
        kwargs (): different keyword arguments

    Returns:
        cuda tensors in its original construct

    """
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(**kwargs)
        except RuntimeError:
            kwargs.pop("non_blocking")
            return obj.to(**kwargs)
    elif isinstance(obj, list):
        return [recursive_to(o, **kwargs) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, set):
        return set(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, **kwargs) for k, v in obj.items()}
    elif hasattr(obj, "to"):
        # this takes care of classes that implements the ~to method
        return obj.to(**kwargs)
    else:
        return obj

from collections.abc import Mapping, Sequence
def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        if isinstance(obj, str):
            return obj
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)
        

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    # Checks
    # print("Mod return", vals, " --> ", retval)
    # if isinstance(retval, torch.Tensor):
    #     notnan_idx = ~torch.isnan(retval)
    #     assert torch.all(retval[notnan_idx] >= range_min)
    #     assert torch.all(retval[notnan_idx] < range_max)
    # else:
    #     assert (
    #         np.nanmin(retval) >= range_min
    #     ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
    #     assert (
    #         np.nanmax(retval) <= range_max
    #     ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
    return retval

def wrapped_mean(x: np.ndarray, axis=None) -> float:
    """
    Wrap the mean function about [-pi, pi]
    """
    # https://rosettacode.org/wiki/Averages/Mean_angle
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    retval = np.arctan2(np.nanmean(sin_x, axis=axis), np.nanmean(cos_x, axis=axis))
    return retval

def tolerant_comparison_check(values, cmp: Literal[">=", "<="], v):
    """
    Compares values in a way that is tolerant of numerical precision
    >>> tolerant_comparison_check(-3.1415927410125732, ">=", -np.pi)
    True
    """
    if cmp == ">=":  # v is a lower bound
        minval = np.nanmin(values)
        diff = minval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True  # Passes
        return diff > 0
    elif cmp == "<=":
        maxval = np.nanmax(values)
        diff = maxval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True
        return diff < 0
    else:
        raise ValueError(f"Illegal comparator: {cmp}")
    
    

# =============================================================================
# Classes
# =============================================================================
# =============================================================================
# Tests
# =============================================================================
if __name__ == "__main__":
    pass
