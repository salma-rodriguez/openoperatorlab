import numpy as np

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.evolution import (
    UnitaryEvolution,
    SemigroupEvolution,
    EvolutionAnalyzer,
)


def test_unitary_evolution_identity_at_zero():
    H = LinearOperator(np.diag([1.0, 2.0]))
    U = UnitaryEvolution(H)

    assert np.allclose(U.matrix(0.0), np.eye(2))


def test_unitary_evolution_diagonal():
    H = LinearOperator(np.diag([1.0, 2.0]))
    U = UnitaryEvolution(H)

    expected = np.diag([
        np.exp(-1j * 1.0),
        np.exp(-1j * 2.0),
    ])

    assert np.allclose(U.matrix(1.0), expected)


def test_unitary_evolution_preserves_norm_for_hermitian_diagonal():
    H = LinearOperator(np.diag([1.0, 2.0]))
    U = UnitaryEvolution(H)

    v = np.array([1.0, 1.0], dtype=complex)
    evolved = U.apply(v, 1.0)

    assert np.isclose(np.linalg.norm(evolved), np.linalg.norm(v))


def test_semigroup_evolution_identity_at_zero():
    A = LinearOperator(np.diag([1.0, 2.0]))
    S = SemigroupEvolution(A)

    assert np.allclose(S.matrix(0.0), np.eye(2))


def test_semigroup_evolution_diagonal():
    A = LinearOperator(np.diag([1.0, 2.0]))
    S = SemigroupEvolution(A)

    expected = np.diag([
        np.exp(1.0),
        np.exp(2.0),
    ])

    assert np.allclose(S.matrix(1.0), expected)


def test_propagator_apply():
    A = LinearOperator(np.diag([1.0, 2.0]))
    S = SemigroupEvolution(A)

    v = np.array([1.0, 1.0])
    expected = np.array([np.exp(1.0), np.exp(2.0)])

    assert np.allclose(S.apply(v, 1.0), expected)


def test_evolution_analyzer_unitary():
    H = LinearOperator(np.diag([1.0, 2.0]))
    E = EvolutionAnalyzer(H)

    v = np.array([1.0, 1.0], dtype=complex)

    assert np.allclose(
        E.propagate_unitary(v, 0.0),
        v,
    )


def test_evolution_analyzer_semigroup():
    A = LinearOperator(np.diag([1.0, 2.0]))
    E = EvolutionAnalyzer(A)

    v = np.array([1.0, 1.0])

    assert np.allclose(
        E.propagate_semigroup(v, 0.0),
        v,
    )
