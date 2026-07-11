from spectral_operators.core.exceptions import (
    DimensionMismatchError,
    InvalidOperatorError,
    NonSquareOperatorError,
    OperatorError,
    SerializationError,
    SingularOperatorError,
)


def test_exception_hierarchy():
    assert issubclass(DimensionMismatchError, OperatorError)
    assert issubclass(NonSquareOperatorError, OperatorError)
    assert issubclass(SingularOperatorError, OperatorError)
    assert issubclass(InvalidOperatorError, OperatorError)
    assert issubclass(SerializationError, OperatorError)


def test_operator_error_message():
    error = OperatorError("operator failed")
    assert str(error) == "operator failed"
