import numpy as np

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.operators import (
    FiniteDifferenceOperator,
    WeightedOperator,
    GradedOperator,
    HamiltonianOperator,
    AdelicOperator,
    ZetaOperator,
)


def test_finite_difference_shape():
    D = FiniteDifferenceOperator(N=8, L=1.0)
    assert D.shape == (8, 8)


def test_finite_difference_first_periodic_is_skew():
    D = FiniteDifferenceOperator(N=8, L=1.0, derivative=1, boundary="periodic")
    assert D.is_skew()


def test_finite_difference_second_periodic_is_symmetric():
    D2 = FiniteDifferenceOperator(N=8, L=1.0, derivative=2, boundary="periodic")
    assert D2.is_symmetric()


def test_weighted_operator_left():
    A = LinearOperator(np.eye(3), name="A")
    w = np.array([1, 2, 3])
    W = WeightedOperator(A, w, side="left")
    assert np.allclose(W.matrix, np.diag(w))


def test_weighted_operator_right():
    A = LinearOperator(np.eye(3), name="A")
    w = np.array([1, 2, 3])
    W = WeightedOperator(A, w, side="right")
    assert np.allclose(W.matrix, np.diag(w))


def test_graded_operator_shape_and_hermitian():
    A = LinearOperator(np.array([[1, 2], [3, 4]], dtype=float), name="A")
    G = GradedOperator(A)
    assert G.shape == (4, 4)
    assert G.is_hermitian()


def test_hamiltonian_operator_enforces_hermitian():
    A = LinearOperator(np.array([[0, 2], [1, 0]], dtype=float), name="A")
    H = HamiltonianOperator(A, enforce_hermitian=True)
    assert H.is_hermitian()


def test_adelic_operator_weighted_sum():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    Ad = AdelicOperator(
        [A, B],
        weights=np.array([1.0, 1.0]),
        normalize=False,
    )

    assert np.allclose(Ad.matrix, 3 * np.eye(2))


def test_adelic_operator_normalized():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(3 * np.eye(2), name="B")

    Ad = AdelicOperator(
        [A, B],
        weights=np.array([1.0, 1.0]),
        normalize=True,
    )

    assert np.allclose(Ad.matrix, 2 * np.eye(2))


def test_zeta_operator_scale_and_shift():
    A = LinearOperator(np.eye(2), name="A")
    Z = ZetaOperator(A, scale=2.0, shift=3.0)

    assert np.allclose(Z.matrix, 5 * np.eye(2))


def test_zeta_operator_without_shift():
    A = LinearOperator(np.array([[1, 2], [3, 4]], dtype=float), name="A")
    Z = ZetaOperator(A, scale=2.0)

    assert np.allclose(Z.matrix, 2 * A.matrix)
