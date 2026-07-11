"""
Spectral operator constructions and analysis tools.

This package provides the public interface for OperatorLab's
spectral-operator functionality.
"""

from .core import (
    DimensionMismatchError,
    Field,
    LinearOperator,
    NonSquareOperatorError,
    Norm,
    OperatorBase,
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
