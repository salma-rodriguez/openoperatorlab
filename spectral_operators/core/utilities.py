"""
spectral_operators.core.utilities
=================================

Shared validation, conversion, and normalization utilities.

These functions support multiple OperatorLab modules without depending
on higher-level operator implementations.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np

from .exceptions import (
    DimensionMismatchError,
    OperatorError,
)


# ===========================================================================
# Array Conversion
# ===========================================================================

def as_array(
    values,
    *,
    name: str = "values",
    dtype=None,
    copy: bool = False,
) -> np.ndarray:
    """
    Convert input values into a NumPy array.

    Parameters
    ----------
    values
        Array-like input.

    name
        Human-readable input name used in error messages.

    dtype
        Optional NumPy dtype.

    copy
        If True, always create an independent array.

    Returns
    -------
    np.ndarray
        Converted array.
    """

    try:
        return np.array(values, dtype=dtype, copy=copy)
    except (TypeError, ValueError) as exc:
        raise OperatorError(
            f"{name} could not be converted to a NumPy array."
        ) from exc


def as_one_dimensional_array(
    values,
    *,
    name: str = "values",
    dtype=None,
    copy: bool = False,
    allow_empty: bool = False,
) -> np.ndarray:
    """
    Convert values into a one-dimensional NumPy array.
    """

    array = as_array(
        values,
        name=name,
        dtype=dtype,
        copy=copy,
    )

    if array.ndim != 1:
        raise OperatorError(
            f"{name} must be one-dimensional."
        )

    if not allow_empty and array.size == 0:
        raise OperatorError(
            f"{name} cannot be empty."
        )

    return array


def as_two_dimensional_array(
    values,
    *,
    name: str = "matrix",
    dtype=None,
    copy: bool = False,
) -> np.ndarray:
    """
    Convert values into a two-dimensional NumPy array.
    """

    array = as_array(
        values,
        name=name,
        dtype=dtype,
        copy=copy,
    )

    if array.ndim != 2:
        raise OperatorError(
            f"{name} must be two-dimensional."
        )

    return array


# ===========================================================================
# Scalar Validation
# ===========================================================================

def require_positive_integer(
    value: Any,
    *,
    name: str = "value",
) -> int:
    """
    Validate and return a positive integer.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise OperatorError(
            f"{name} must be an integer."
        )

    value = int(value)

    if value < 1:
        raise OperatorError(
            f"{name} must be positive."
        )

    return value


def require_nonnegative_integer(
    value: Any,
    *,
    name: str = "value",
) -> int:
    """
    Validate and return a nonnegative integer.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise OperatorError(
            f"{name} must be an integer."
        )

    value = int(value)

    if value < 0:
        raise OperatorError(
            f"{name} must be nonnegative."
        )

    return value


def require_positive_real(
    value: Any,
    *,
    name: str = "value",
) -> float:
    """
    Validate and return a positive real number.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise OperatorError(
            f"{name} must be a real number."
        )

    value = float(value)

    if not np.isfinite(value):
        raise OperatorError(
            f"{name} must be finite."
        )

    if value <= 0:
        raise OperatorError(
            f"{name} must be positive."
        )

    return value


def require_probability(
    value: Any,
    *,
    name: str = "value",
    inclusive: bool = True,
) -> float:
    """
    Validate a probability-like scalar.

    With ``inclusive=True``, the accepted interval is [0, 1].
    Otherwise, the accepted interval is (0, 1).
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise OperatorError(
            f"{name} must be a real number."
        )

    value = float(value)

    if not np.isfinite(value):
        raise OperatorError(
            f"{name} must be finite."
        )

    if inclusive:
        valid = 0.0 <= value <= 1.0
        interval = "[0, 1]"
    else:
        valid = 0.0 < value < 1.0
        interval = "(0, 1)"

    if not valid:
        raise OperatorError(
            f"{name} must lie in {interval}."
        )

    return value


# ===========================================================================
# Shape Validation
# ===========================================================================

def require_square_shape(
    shape,
    *,
    name: str = "shape",
) -> tuple[int, int]:
    """
    Validate a square two-dimensional shape.
    """

    if len(shape) != 2:
        raise OperatorError(
            f"{name} must contain exactly two dimensions."
        )

    rows, cols = shape

    if rows != cols:
        raise OperatorError(
            f"{name} must be square."
        )

    return int(rows), int(cols)


def require_same_shape(
    left,
    right,
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[int, ...]:
    """
    Validate that two arrays or shape-like objects have equal shapes.
    """

    left_shape = left.shape if hasattr(left, "shape") else tuple(left)
    right_shape = right.shape if hasattr(right, "shape") else tuple(right)

    if left_shape != right_shape:
        raise DimensionMismatchError(
            f"{left_name} shape {left_shape} does not match "
            f"{right_name} shape {right_shape}."
        )

    return tuple(left_shape)


def require_matmul_compatible(
    left,
    right,
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    """
    Validate matrix-multiplication compatibility.
    """

    left_shape = left.shape if hasattr(left, "shape") else tuple(left)
    right_shape = right.shape if hasattr(right, "shape") else tuple(right)

    if len(left_shape) != 2 or len(right_shape) != 2:
        raise OperatorError(
            "Matrix multiplication requires two-dimensional operands."
        )

    if left_shape[1] != right_shape[0]:
        raise DimensionMismatchError(
            f"{left_name} shape {left_shape} is incompatible with "
            f"{right_name} shape {right_shape}."
        )


# ===========================================================================
# Normalization
# ===========================================================================

def normalize_l1(
    values,
    *,
    name: str = "values",
    dtype=None,
) -> np.ndarray:
    """
    Normalize values by their absolute sum.

    Computes

        values / sum(abs(values)).
    """

    array = as_one_dimensional_array(
        values,
        name=name,
        dtype=dtype,
        copy=True,
    )

    total = np.sum(np.abs(array))

    if total == 0:
        raise OperatorError(
            f"{name} cannot be L1-normalized because its absolute sum is zero."
        )

    return array / total


def normalize_max_abs(
    values,
    *,
    name: str = "values",
    dtype=None,
) -> np.ndarray:
    """
    Normalize values by their maximum absolute value.
    """

    array = as_array(
        values,
        name=name,
        dtype=dtype,
        copy=True,
    )

    if array.size == 0:
        raise OperatorError(
            f"{name} cannot be empty."
        )

    maximum = np.max(np.abs(array))

    if maximum == 0:
        raise OperatorError(
            f"{name} cannot be normalized because its maximum absolute value is zero."
        )

    return array / maximum


# ===========================================================================
# Immutability Helpers
# ===========================================================================

def readonly_array(
    values,
    *,
    name: str = "values",
    dtype=None,
    ndim: int | None = None,
) -> np.ndarray:
    """
    Return an independent, read-only NumPy array.
    """

    array = as_array(
        values,
        name=name,
        dtype=dtype,
        copy=True,
    )

    if ndim is not None and array.ndim != ndim:
        raise OperatorError(
            f"{name} must have exactly {ndim} dimensions."
        )

    array.setflags(write=False)

    return array
