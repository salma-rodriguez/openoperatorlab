import numpy as np

from spectral_operators import LinearOperator, Norm, OperatorFactory


def test_transpose():
    A = LinearOperator(np.array([[1, 2], [3, 4]]))
    assert np.allclose(A.transpose.matrix, np.array([[1, 3], [2, 4]]))


def test_adjoint_complex():
    A = LinearOperator(np.array([[1+1j, 2], [3, 4-2j]]))
    expected = np.array([[1-1j, 3], [2, 4+2j]])
    assert np.allclose(A.adjoint.matrix, expected)


def test_arithmetic_addition():
    A = LinearOperator(np.eye(2))
    B = LinearOperator(2 * np.eye(2))
    C = A + B
    assert np.allclose(C.matrix, 3 * np.eye(2))


def test_matmul():
    A = LinearOperator(np.array([[1, 2], [3, 4]]))
    B = LinearOperator(np.eye(2))
    C = A @ B
    assert np.allclose(C.matrix, A.matrix)


def test_norms():
    A = LinearOperator(np.array([[3, 4], [0, 0]]))
    assert np.isclose(A.norm("fro"), 5.0)
    assert np.isclose(A.norm(Norm.FROBENIUS), 5.0)


def test_symmetric_decomposition_reconstructs():
    A = LinearOperator(np.array([[1, 2], [3, 4]], dtype=float))
    R = A.symmetric_part() + A.skew_part()
    assert np.allclose(R.matrix, A.matrix)


def test_hermitian_decomposition_reconstructs():
    A = LinearOperator(np.array([[1+1j, 2], [3, 4-2j]]))
    R = A.hermitian_part() + A.antihermitian_part()
    assert np.allclose(R.matrix, A.matrix)


def test_eigendecomposition():
    A = LinearOperator(np.diag([2, 3]))
    vals, vecs = A.eigendecomposition()
    assert np.allclose(np.sort(vals), np.array([2, 3]))
    assert vecs.shape == (2, 2)


def test_factory_identity():
    A = OperatorFactory.identity(3)
    assert isinstance(A, LinearOperator)
    assert np.allclose(A.matrix, np.eye(3))


def test_factory_zeros():
    A = OperatorFactory.zeros((2, 3))
    assert A.shape == (2, 3)
    assert np.allclose(A.matrix, np.zeros((2, 3)))


def test_factory_ones():
    A = OperatorFactory.ones((2, 2))
    assert np.allclose(A.matrix, np.ones((2, 2)))


def test_factory_diagonal():
    A = OperatorFactory.diagonal([1, 2, 3])
    assert np.allclose(A.matrix, np.diag([1, 2, 3]))


def test_factory_random_reproducible():
    A = OperatorFactory.random((2, 2), seed=42)
    B = OperatorFactory.random((2, 2), seed=42)
    assert np.allclose(A.matrix, B.matrix)


def test_factory_random_complex():
    A = OperatorFactory.random((2, 2), dtype=complex, seed=42)
    assert np.iscomplexobj(A.matrix)


def test_factory_block():
    A = OperatorFactory.identity(2)
    Z = OperatorFactory.zeros((2, 2))
    B = OperatorFactory.block([[A, Z], [Z, A]])

    expected = np.block([
        [np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), np.eye(2)]
    ])

    assert np.allclose(B.matrix, expected)
