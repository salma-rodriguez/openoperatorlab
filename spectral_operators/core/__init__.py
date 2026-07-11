"""
Core abstractions for the spectral_operators package.
"""

from .exceptions import (
    DimensionMismatchError,
    InvalidOperatorError,
    NonSquareOperatorError,
    OperatorError,
    SerializationError,
    SingularOperatorError,
)

from .base import OperatorBase

from .utilities import (
    as_array,
    as_one_dimensional_array,
    as_two_dimensional_array,
    normalize_l1,
    normalize_max_abs,
    readonly_array,
    require_matmul_compatible,
    require_nonnegative_integer,
    require_positive_integer,
    require_positive_real,
    require_probability,
    require_same_shape,
    require_square_shape,
)

from .algebra import (
    DimensionMismatchError,
    Field,
    LinearOperator,
    NonSquareOperatorError,
    Norm,
    OperatorError,
    OperatorFactory,
    SingularOperatorError,
)

__all__ = [
    "DimensionMismatchError",
    "Field",
    "LinearOperator",
    "NonSquareOperatorError",
    "Norm",
    "OperatorBase",
    "OperatorError",
    "OperatorFactory",
    "SingularOperatorError",
]
