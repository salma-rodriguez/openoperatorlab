# spectral_operators/core/exceptions.py

"""
Exception hierarchy for OperatorLab.
"""


class OperatorError(Exception):
    """Base exception for operator-related errors."""


class DimensionMismatchError(OperatorError):
    """Raised when operator dimensions are incompatible."""


class NonSquareOperatorError(OperatorError):
    """Raised when an operation requires a square operator."""


class SingularOperatorError(OperatorError):
    """Raised when an operator is singular or non-invertible."""


class InvalidOperatorError(OperatorError):
    """Raised when an invalid operator object is supplied."""


class SerializationError(OperatorError):
    """Raised when serialization or deserialization fails."""
