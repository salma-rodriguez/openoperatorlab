"""
Tests for stochastic_operators.stationary.
"""

import numpy as np
import pytest

from operator_core import (
    DimensionMismatchError,
    OperatorError,
)
from stochastic_operators import (
    MarkovGenerator,
    MarkovOperator,
    StationaryAnalyzer,
    StochasticOperator,
)


# ===========================================================================
# Construction and Model Detection
# ===========================================================================

def test_stationary_analyzer_accepts_markov_operator():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    analyzer = StationaryAnalyzer(
        operator
    )

    assert analyzer.operator is operator
    assert analyzer.dimension == 2
    assert analyzer.states == (0, 1)
    assert analyzer.is_discrete_time
    assert not analyzer.is_continuous_time


def test_stationary_analyzer_accepts_markov_generator():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    analyzer = StationaryAnalyzer(
        generator
    )

    assert analyzer.operator is generator
    assert analyzer.dimension == 2
    assert analyzer.states == (0, 1)
    assert analyzer.is_continuous_time
    assert not analyzer.is_discrete_time


def test_stationary_analyzer_rejects_plain_stochastic_operator():
    operator = StochasticOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    with pytest.raises(OperatorError):
        StationaryAnalyzer(operator)


def test_stationary_analyzer_rejects_invalid_input():
    with pytest.raises(OperatorError):
        StationaryAnalyzer(
            np.eye(2)
        )


def test_stationary_analyzer_uses_operator_tolerance():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        tol=1e-8,
    )

    analyzer = StationaryAnalyzer(
        operator
    )

    assert np.isclose(
        analyzer.tol,
        1e-8,
    )


def test_stationary_analyzer_accepts_custom_tolerance():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    analyzer = StationaryAnalyzer(
        operator,
        tol=1e-7,
    )

    assert np.isclose(
        analyzer.tol,
        1e-7,
    )


def test_stationary_analyzer_rejects_invalid_tolerance():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    with pytest.raises(OperatorError):
        StationaryAnalyzer(
            operator,
            tol=-1e-8,
        )

    with pytest.raises(OperatorError):
        StationaryAnalyzer(
            operator,
            tol=np.inf,
        )

    with pytest.raises(OperatorError):
        StationaryAnalyzer(
            operator,
            tol=[1e-8],
        )


# ===========================================================================
# Stationary System Matrices
# ===========================================================================

def test_row_markov_stationary_system():
    matrix = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    analyzer = StationaryAnalyzer(
        MarkovOperator(matrix)
    )

    expected = (
        matrix.T
        - np.eye(2)
    )

    assert np.allclose(
        analyzer.stationary_system_matrix(),
        expected,
    )


def test_column_markov_stationary_system():
    matrix = np.array([
        [0.8, 0.3],
        [0.2, 0.7],
    ])

    analyzer = StationaryAnalyzer(
        MarkovOperator(
            matrix,
            convention="column",
        )
    )

    expected = (
        matrix
        - np.eye(2)
    )

    assert np.allclose(
        analyzer.stationary_system_matrix(),
        expected,
    )


def test_row_generator_stationary_system():
    matrix = np.array([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    analyzer = StationaryAnalyzer(
        MarkovGenerator(matrix)
    )

    assert np.allclose(
        analyzer.stationary_system_matrix(),
        matrix.T,
    )


def test_column_generator_stationary_system():
    matrix = np.array([
        [-0.3, 0.1],
        [0.3, -0.1],
    ])

    analyzer = StationaryAnalyzer(
        MarkovGenerator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.stationary_system_matrix(),
        matrix,
    )


def test_stationary_system_matrix_is_read_only():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    matrix = (
        analyzer.stationary_system_matrix()
    )

    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0


# ===========================================================================
# Stationary Space
# ===========================================================================

def test_unique_stationary_space_dimension():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    assert (
        analyzer.stationary_dimension()
        == 1
    )
    assert analyzer.is_unique()


def test_nonunique_stationary_space_dimension():
    analyzer = StationaryAnalyzer(
        MarkovOperator(
            np.eye(3)
        )
    )

    assert (
        analyzer.stationary_dimension()
        == 3
    )
    assert not analyzer.is_unique()


def test_stationary_space_basis_shape():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    basis = (
        analyzer.stationary_space_basis()
    )

    assert basis.shape == (2, 1)


def test_stationary_space_basis_is_read_only():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    basis = (
        analyzer.stationary_space_basis()
    )

    with pytest.raises(ValueError):
        basis[0, 0] = 0.0


# ===========================================================================
# Stationary Distributions
# ===========================================================================

def test_row_markov_stationary_distribution():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert np.allclose(
        distribution,
        [0.6, 0.4],
        atol=1e-9,
    )
    assert analyzer.is_stationary(
        distribution
    )


def test_column_markov_stationary_distribution():
    analyzer = StationaryAnalyzer(
        MarkovOperator(
            [
                [0.8, 0.3],
                [0.2, 0.7],
            ],
            convention="column",
        )
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert np.allclose(
        distribution,
        [0.6, 0.4],
        atol=1e-9,
    )


def test_row_generator_stationary_distribution():
    analyzer = StationaryAnalyzer(
        MarkovGenerator([
            [-0.3, 0.3],
            [0.1, -0.1],
        ])
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert np.allclose(
        distribution,
        [0.25, 0.75],
        atol=1e-9,
    )


def test_column_generator_stationary_distribution():
    analyzer = StationaryAnalyzer(
        MarkovGenerator(
            [
                [-0.3, 0.1],
                [0.3, -0.1],
            ],
            convention="column",
        )
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert np.allclose(
        distribution,
        [0.25, 0.75],
        atol=1e-9,
    )


def test_reducible_chain_returns_feasible_distribution():
    analyzer = StationaryAnalyzer(
        MarkovOperator(
            np.eye(3)
        )
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert np.all(
        distribution >= 0.0
    )
    assert np.isclose(
        distribution.sum(),
        1.0,
    )
    assert analyzer.is_stationary(
        distribution
    )
    assert not analyzer.is_unique()


def test_stationary_distribution_is_read_only():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    with pytest.raises(ValueError):
        distribution[0] = 1.0


# ===========================================================================
# Residuals and Validation
# ===========================================================================

def test_stationarity_residual_is_small():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    residual = (
        analyzer.stationarity_residual(
            [0.6, 0.4]
        )
    )

    assert residual < 1e-12


def test_nonstationary_distribution_has_positive_residual():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    residual = (
        analyzer.stationarity_residual(
            [1.0, 0.0]
        )
    )

    assert residual > 0.0
    assert not analyzer.is_stationary(
        [1.0, 0.0]
    )


def test_invalid_distribution_dimension_rejected():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    with pytest.raises(
        DimensionMismatchError
    ):
        analyzer.stationarity_residual(
            [0.2, 0.3, 0.5]
        )


def test_multidimensional_distribution_rejected():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    with pytest.raises(OperatorError):
        analyzer.stationarity_residual(
            [[0.6, 0.4]]
        )


def test_negative_distribution_rejected():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    with pytest.raises(OperatorError):
        analyzer.stationarity_residual(
            [1.1, -0.1]
        )


def test_distribution_sum_rejected():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    with pytest.raises(OperatorError):
        analyzer.stationarity_residual(
            [0.8, 0.8]
        )


def test_is_stationary_returns_false_for_invalid_distribution():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    assert not analyzer.is_stationary(
        [0.8, 0.8]
    )
    assert not analyzer.is_stationary(
        [1.0, 0.0, 0.0]
    )


# ===========================================================================
# Row-Oriented Representations
# ===========================================================================

def test_row_oriented_row_markov_matrix():
    matrix = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    analyzer = StationaryAnalyzer(
        MarkovOperator(matrix)
    )

    assert np.allclose(
        analyzer.row_oriented_matrix(),
        matrix,
    )


def test_row_oriented_column_markov_matrix():
    matrix = np.array([
        [0.8, 0.3],
        [0.2, 0.7],
    ])

    analyzer = StationaryAnalyzer(
        MarkovOperator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.row_oriented_matrix(),
        matrix.T,
    )


def test_row_oriented_column_generator_matrix():
    matrix = np.array([
        [-0.3, 0.1],
        [0.3, -0.1],
    ])

    analyzer = StationaryAnalyzer(
        MarkovGenerator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.row_oriented_matrix(),
        matrix.T,
    )


def test_row_oriented_matrix_is_read_only():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    matrix = (
        analyzer.row_oriented_matrix()
    )

    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0


# ===========================================================================
# Detailed Balance and Reversibility
# ===========================================================================

def test_reversible_markov_chain():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert analyzer.is_reversible(
        distribution
    )
    assert (
        analyzer.detailed_balance_defect(
            distribution
        )
        < 1e-12
    )


def test_reversible_markov_generator():
    analyzer = StationaryAnalyzer(
        MarkovGenerator([
            [-0.3, 0.3],
            [0.1, -0.1],
        ])
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert analyzer.is_reversible(
        distribution
    )
    assert (
        analyzer.detailed_balance_defect(
            distribution
        )
        < 1e-12
    )


def test_nonreversible_three_state_cycle():
    operator = MarkovOperator([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])

    analyzer = StationaryAnalyzer(
        operator
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    assert np.allclose(
        distribution,
        [1 / 3, 1 / 3, 1 / 3],
        atol=1e-9,
    )
    assert not analyzer.is_reversible(
        distribution
    )
    assert (
        analyzer.detailed_balance_defect(
            distribution
        )
        > 0.0
    )


def test_detailed_balance_matrix_is_antisymmetric():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
    )

    defect = (
        analyzer.detailed_balance_matrix()
    )

    assert np.allclose(
        defect,
        -defect.T,
    )


def test_detailed_balance_matrix_is_read_only():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    defect = (
        analyzer.detailed_balance_matrix()
    )

    with pytest.raises(ValueError):
        defect[0, 1] = 0.0


def test_detailed_balance_accepts_explicit_distribution():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    defect = (
        analyzer.detailed_balance_defect(
            [0.6, 0.4]
        )
    )

    assert defect < 1e-12


# ===========================================================================
# Long-Time Diagnostics
# ===========================================================================

def test_discrete_time_limiting_distribution():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    limiting = (
        analyzer.limiting_distribution(
            [1.0, 0.0],
            steps=100,
        )
    )

    assert np.allclose(
        limiting,
        [0.6, 0.4],
        atol=1e-8,
    )


def test_continuous_time_limiting_distribution():
    analyzer = StationaryAnalyzer(
        MarkovGenerator([
            [-0.3, 0.3],
            [0.1, -0.1],
        ])
    )

    limiting = (
        analyzer.limiting_distribution(
            [1.0, 0.0],
            time=100.0,
        )
    )

    assert np.allclose(
        limiting,
        [0.25, 0.75],
        atol=1e-8,
    )


def test_discrete_time_limiting_error_is_small():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
    )

    error = analyzer.limiting_error(
        [1.0, 0.0],
        steps=100,
    )

    assert error < 1e-8


def test_continuous_time_limiting_error_is_small():
    analyzer = StationaryAnalyzer(
        MarkovGenerator([
            [-0.3, 0.3],
            [0.1, -0.1],
        ])
    )

    error = analyzer.limiting_error(
        [1.0, 0.0],
        time=100.0,
    )

    assert error < 1e-8


def test_periodic_chain_need_not_converge():
    analyzer = StationaryAnalyzer(
        MarkovOperator([
            [0.0, 1.0],
            [1.0, 0.0],
        ])
    )

    limiting = (
        analyzer.limiting_distribution(
            [1.0, 0.0],
            steps=101,
        )
    )

    assert np.allclose(
        limiting,
        [0.0, 1.0],
    )

    assert (
        analyzer.limiting_error(
            [1.0, 0.0],
            steps=101,
        )
        > 0.0
    )


# ===========================================================================
# Summary
# ===========================================================================

def test_stationary_summary_discrete_time():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        states=("a", "b"),
        name="P",
    )

    summary = StationaryAnalyzer(
        operator
    ).summary()

    assert summary["operator"] == "P"
    assert (
        summary["model_type"]
        == "discrete_time"
    )
    assert summary["dimension"] == 2
    assert summary["states"] == ("a", "b")
    assert np.allclose(
        summary[
            "stationary_distribution"
        ],
        (0.6, 0.4),
        atol=1e-9,
    )
    assert summary[
        "stationary_dimension"
    ] == 1
    assert summary["unique"]
    assert (
        summary[
            "stationarity_residual"
        ]
        < 1e-12
    )
    assert summary["reversible"]


def test_stationary_summary_continuous_time():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
        name="Q",
    )

    summary = StationaryAnalyzer(
        generator
    ).summary()

    assert summary["operator"] == "Q"
    assert (
        summary["model_type"]
        == "continuous_time"
    )
    assert summary["dimension"] == 2
    assert summary["states"] == ("a", "b")
    assert np.allclose(
        summary[
            "stationary_distribution"
        ],
        (0.25, 0.75),
        atol=1e-9,
    )
    assert summary[
        "stationary_dimension"
    ] == 1
    assert summary["unique"]
    assert summary["reversible"]
