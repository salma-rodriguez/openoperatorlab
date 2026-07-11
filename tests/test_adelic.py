import numpy as np

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.operators import AdelicOperator
from spectral_operators.adelic import (
    LocalComponent,
    AdelicSystem,
    AdelicBuilder,
    AdelicAnalyzer,
)


def test_local_component_basic():
    A = LinearOperator(np.eye(2), name="A")
    c = LocalComponent(2, A)

    assert c.label == 2
    assert c.shape == (2, 2)
    assert np.allclose(c.matrix, np.eye(2))
    assert c.as_tuple() == (2, A)


def test_adelic_system_from_components():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    system = AdelicSystem([
        LocalComponent(2, A),
        LocalComponent(3, B),
    ])

    assert system.labels == (2, 3)
    assert system.shape == (2, 2)
    assert len(system.components) == 2


def test_adelic_system_from_operators():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    system = AdelicSystem.from_operators(
        [A, B],
        labels=[2, 3],
        weights=[1, 1],
        normalize=True,
    )

    assert system.labels == (2, 3)
    assert np.allclose(system.weight_array(), np.array([0.5, 0.5]))


def test_adelic_builder_builds_operator():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    system = AdelicSystem.from_operators(
        [A, B],
        labels=[2, 3],
        weights=[1, 1],
        normalize=False,
    )

    Ad = AdelicBuilder(system).build()

    assert isinstance(Ad, AdelicOperator)
    assert np.allclose(Ad.matrix, 3 * np.eye(2))


def test_adelic_builder_builds_normalized_operator():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(3 * np.eye(2), name="B")

    system = AdelicSystem.from_operators(
        [A, B],
        labels=[2, 3],
        weights=[1, 1],
        normalize=True,
    )

    Ad = AdelicBuilder(system).build()

    assert np.allclose(Ad.matrix, 2 * np.eye(2))


def test_adelic_analyzer_component_norms():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    system = AdelicSystem.from_operators(
        [A, B],
        labels=[2, 3],
        weights=[1, 1],
        normalize=False,
    )

    analyzer = AdelicAnalyzer(system)
    norms = analyzer.component_norms()

    assert np.isclose(norms[2], np.sqrt(2))
    assert np.isclose(norms[3], 2 * np.sqrt(2))


def test_adelic_analyzer_weighted_component_norms():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    system = AdelicSystem.from_operators(
        [A, B],
        labels=[2, 3],
        weights=[0.25, 0.75],
        normalize=False,
    )

    analyzer = AdelicAnalyzer(system)
    norms = analyzer.weighted_component_norms()

    assert np.isclose(norms[2], 0.25 * np.sqrt(2))
    assert np.isclose(norms[3], 0.75 * 2 * np.sqrt(2))


def test_adelic_analyzer_weight_summary():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    system = AdelicSystem.from_operators(
        [A, B],
        labels=[2, 3],
        weights=[0.25, 0.75],
        normalize=False,
    )

    summary = AdelicAnalyzer(system).weight_summary()

    assert summary["labels"] == (2, 3)
    assert summary["weights"] == (0.25, 0.75)
    assert np.isclose(summary["sum_abs_weights"], 1.0)
    assert summary["num_components"] == 2


def test_adelic_analyzer_summary():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    system = AdelicSystem.from_operators(
        [A, B],
        labels=[2, 3],
        weights=[1, 1],
        normalize=True,
    )

    summary = AdelicAnalyzer(system).summary()

    assert summary["shape"] == (2, 2)
    assert summary["labels"] == (2, 3)
    assert "component_norms" in summary
    assert "weighted_component_norms" in summary
    assert "weights" in summary
