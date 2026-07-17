"""
Tests for stochastic_operators.ergodic.
"""

import numpy as np
import pytest

from operator_core import (
    DimensionMismatchError,
    OperatorError,
)
from stochastic_operators import (
    ErgodicAnalyzer,
    MarkovGenerator,
    MarkovOperator,
    StochasticOperator,
)


# ===========================================================================
# Shared Test Models
# ===========================================================================

def make_ergodic_operator():
    """
    Return an irreducible, aperiodic two-state Markov operator.
    """

    return MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])


def make_periodic_operator():
    """
    Return an irreducible Markov operator with period two.
    """

    return MarkovOperator([
        [0.0, 1.0],
        [1.0, 0.0],
    ])


def make_reducible_operator():
    """
    Return a reducible Markov operator with two closed classes.
    """

    return MarkovOperator([
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, 0.5],
    ])


def make_irreducible_generator():
    """
    Return an irreducible two-state continuous-time generator.
    """

    return MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])


def make_reducible_generator():
    """
    Return a reducible continuous-time generator.
    """

    return MarkovGenerator([
        [0.0, 0.0, 0.0],
        [0.0, -0.4, 0.4],
        [0.0, 0.2, -0.2],
    ])


# ===========================================================================
# Construction and Model Detection
# ===========================================================================

def test_ergodic_analyzer_accepts_markov_operator():
    operator = make_ergodic_operator()

    analyzer = ErgodicAnalyzer(
        operator
    )

    assert analyzer.model is operator
    assert analyzer.dimension == 2
    assert analyzer.states == (0, 1)
    assert analyzer.is_discrete_time
    assert not analyzer.is_continuous_time


def test_ergodic_analyzer_accepts_markov_generator():
    generator = make_irreducible_generator()

    analyzer = ErgodicAnalyzer(
        generator
    )

    assert analyzer.model is generator
    assert analyzer.dimension == 2
    assert analyzer.states == (0, 1)
    assert analyzer.is_continuous_time
    assert not analyzer.is_discrete_time


def test_ergodic_analyzer_preserves_state_labels():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        states=("a", "b"),
    )

    analyzer = ErgodicAnalyzer(
        operator
    )

    assert analyzer.states == ("a", "b")


def test_ergodic_analyzer_rejects_plain_stochastic_operator():
    operator = StochasticOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    with pytest.raises(OperatorError):
        ErgodicAnalyzer(operator)


def test_ergodic_analyzer_rejects_array():
    with pytest.raises(OperatorError):
        ErgodicAnalyzer(
            np.eye(2)
        )


def test_ergodic_analyzer_uses_model_tolerance():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        tol=1e-8,
    )

    analyzer = ErgodicAnalyzer(
        operator
    )

    assert np.isclose(
        analyzer.tol,
        1e-8,
    )


def test_ergodic_analyzer_accepts_custom_tolerance():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator(),
        tol=1e-7,
    )

    assert np.isclose(
        analyzer.tol,
        1e-7,
    )


@pytest.mark.parametrize(
    "tol",
    [
        -1e-8,
        np.inf,
        -np.inf,
        np.nan,
        True,
        [1e-8],
    ],
)
def test_ergodic_analyzer_rejects_invalid_tolerance(
    tol,
):
    with pytest.raises(OperatorError):
        ErgodicAnalyzer(
            make_ergodic_operator(),
            tol=tol,
        )


def test_stationary_analyzer_matches_model():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator(),
        tol=1e-7,
    )

    stationary = (
        analyzer.stationary_analyzer
    )

    assert (
        stationary.operator
        is analyzer.model
    )
    assert np.isclose(
        stationary.tol,
        analyzer.tol,
    )


# ===========================================================================
# Row-Oriented Matrices
# ===========================================================================

def test_row_oriented_row_markov_matrix():
    matrix = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    analyzer = ErgodicAnalyzer(
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

    analyzer = ErgodicAnalyzer(
        MarkovOperator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.row_oriented_matrix(),
        matrix.T,
    )


def test_row_oriented_row_generator_matrix():
    matrix = np.array([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    analyzer = ErgodicAnalyzer(
        MarkovGenerator(matrix)
    )

    assert np.allclose(
        analyzer.row_oriented_matrix(),
        matrix,
    )


def test_row_oriented_column_generator_matrix():
    matrix = np.array([
        [-0.3, 0.1],
        [0.3, -0.1],
    ])

    analyzer = ErgodicAnalyzer(
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
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    matrix = (
        analyzer.row_oriented_matrix()
    )

    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0


# ===========================================================================
# Adjacency Matrices
# ===========================================================================

def test_discrete_time_adjacency_matrix():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
        ])
    )

    expected = np.array([
        [True, True, False],
        [False, True, True],
        [True, False, False],
    ])

    assert np.array_equal(
        analyzer.adjacency_matrix(),
        expected,
    )


def test_generator_adjacency_excludes_diagonal():
    analyzer = ErgodicAnalyzer(
        MarkovGenerator([
            [-0.3, 0.3],
            [0.1, -0.1],
        ])
    )

    expected = np.array([
        [False, True],
        [True, False],
    ])

    assert np.array_equal(
        analyzer.adjacency_matrix(),
        expected,
    )


def test_adjacency_respects_tolerance():
    operator = MarkovOperator(
        [
            [1.0 - 1e-10, 1e-10],
            [0.5, 0.5],
        ],
        tol=1e-12,
    )

    analyzer = ErgodicAnalyzer(
        operator,
        tol=1e-8,
    )

    expected = np.array([
        [True, False],
        [True, True],
    ])

    assert np.array_equal(
        analyzer.adjacency_matrix(),
        expected,
    )


def test_adjacency_matrix_is_boolean():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert (
        analyzer.adjacency_matrix().dtype
        == np.dtype(bool)
    )


def test_adjacency_matrix_is_read_only():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    adjacency = (
        analyzer.adjacency_matrix()
    )

    with pytest.raises(ValueError):
        adjacency[0, 0] = False


# ===========================================================================
# Communicating Classes and Irreducibility
# ===========================================================================

def test_irreducible_chain_has_one_communicating_class():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert (
        analyzer.communicating_classes()
        == ((0, 1),)
    )
    assert analyzer.is_irreducible()


def test_irreducible_generator_has_one_communicating_class():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    assert (
        analyzer.communicating_classes()
        == ((0, 1),)
    )
    assert analyzer.is_irreducible()


def test_reducible_chain_has_multiple_communicating_classes():
    analyzer = ErgodicAnalyzer(
        make_reducible_operator()
    )

    assert (
        analyzer.communicating_classes()
        == ((0,), (1, 2))
    )
    assert not analyzer.is_irreducible()


def test_reducible_generator_has_multiple_communicating_classes():
    analyzer = ErgodicAnalyzer(
        make_reducible_generator()
    )

    assert (
        analyzer.communicating_classes()
        == ((0,), (1, 2))
    )
    assert not analyzer.is_irreducible()


def test_communicating_classes_return_state_labels():
    operator = MarkovOperator(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5],
        ],
        states=("absorbing", "left", "right"),
    )

    analyzer = ErgodicAnalyzer(
        operator
    )

    assert (
        analyzer.communicating_classes()
        == (
            ("absorbing",),
            ("left", "right"),
        )
    )


def test_identity_chain_has_singleton_classes():
    analyzer = ErgodicAnalyzer(
        MarkovOperator(
            np.eye(3)
        )
    )

    assert (
        analyzer.communicating_classes()
        == ((0,), (1,), (2,))
    )
    assert not analyzer.is_irreducible()


def test_one_state_chain_is_irreducible():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [1.0],
        ])
    )

    assert (
        analyzer.communicating_classes()
        == ((0,),)
    )
    assert analyzer.is_irreducible()


# ===========================================================================
# Periodicity and Aperiodicity
# ===========================================================================

def test_two_state_alternating_chain_has_period_two():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    assert analyzer.period() == 2
    assert not analyzer.is_aperiodic()


def test_three_state_cycle_has_period_three():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
    )

    assert analyzer.period() == 3
    assert not analyzer.is_aperiodic()


def test_self_loop_makes_irreducible_chain_aperiodic():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert analyzer.period() == 1
    assert analyzer.is_aperiodic()


def test_irreducible_chain_with_only_even_cycles_has_period_two():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
    )

    assert analyzer.period() == 2
    assert not analyzer.is_aperiodic()


def test_irreducible_chain_with_two_and_three_cycles_is_aperiodic():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
    )

    assert analyzer.period() == 1
    assert analyzer.is_aperiodic()


def test_reducible_discrete_chain_has_no_global_period():
    analyzer = ErgodicAnalyzer(
        make_reducible_operator()
    )

    assert analyzer.period() is None
    assert not analyzer.is_aperiodic()


def test_continuous_time_generator_has_no_discrete_period():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    assert analyzer.period() is None
    assert analyzer.is_aperiodic()


def test_reducible_generator_is_still_aperiodic_in_semigroup_sense():
    analyzer = ErgodicAnalyzer(
        make_reducible_generator()
    )

    assert analyzer.period() is None
    assert analyzer.is_aperiodic()


def test_one_state_chain_has_period_one():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [1.0],
        ])
    )

    assert analyzer.period() == 1
    assert analyzer.is_aperiodic()


# ===========================================================================
# Ergodicity Classification
# ===========================================================================

def test_irreducible_aperiodic_chain_is_ergodic():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert analyzer.is_irreducible()
    assert analyzer.is_aperiodic()
    assert analyzer.is_ergodic()


def test_irreducible_periodic_chain_is_not_ergodic():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    assert analyzer.is_irreducible()
    assert not analyzer.is_aperiodic()
    assert not analyzer.is_ergodic()


def test_reducible_aperiodic_chain_is_not_ergodic():
    analyzer = ErgodicAnalyzer(
        make_reducible_operator()
    )

    assert not analyzer.is_irreducible()
    assert not analyzer.is_ergodic()


def test_irreducible_generator_is_ergodic():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    assert analyzer.is_irreducible()
    assert analyzer.is_aperiodic()
    assert analyzer.is_ergodic()


def test_reducible_generator_is_not_ergodic():
    analyzer = ErgodicAnalyzer(
        make_reducible_generator()
    )

    assert not analyzer.is_irreducible()
    assert not analyzer.is_ergodic()


def test_one_state_chain_is_ergodic():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [1.0],
        ])
    )

    assert analyzer.is_ergodic()


# ===========================================================================
# Eigenvalues and Spectral Gaps
# ===========================================================================

def test_discrete_time_eigenvalues():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    eigenvalues = np.sort_complex(
        analyzer.eigenvalues()
    )

    expected = np.sort_complex(
        np.array([
            1.0,
            0.5,
        ])
    )

    assert np.allclose(
        eigenvalues,
        expected,
    )


def test_generator_eigenvalues():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    eigenvalues = np.sort_complex(
        analyzer.eigenvalues()
    )

    expected = np.sort_complex(
        np.array([
            0.0,
            -0.4,
        ])
    )

    assert np.allclose(
        eigenvalues,
        expected,
    )


def test_eigenvalues_are_read_only():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    eigenvalues = (
        analyzer.eigenvalues()
    )

    with pytest.raises(ValueError):
        eigenvalues[0] = 0.0


def test_discrete_time_spectral_gap():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert np.isclose(
        analyzer.spectral_gap(),
        0.5,
    )
    assert (
        analyzer.has_positive_spectral_gap()
    )


def test_continuous_time_spectral_gap():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    assert np.isclose(
        analyzer.spectral_gap(),
        0.4,
    )
    assert (
        analyzer.has_positive_spectral_gap()
    )


def test_periodic_chain_has_zero_absolute_spectral_gap():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    assert np.isclose(
        analyzer.spectral_gap(),
        0.0,
        atol=1e-12,
    )
    assert not (
        analyzer.has_positive_spectral_gap()
    )


def test_identity_chain_has_zero_spectral_gap():
    analyzer = ErgodicAnalyzer(
        MarkovOperator(
            np.eye(3)
        )
    )

    assert np.isclose(
        analyzer.spectral_gap(),
        0.0,
    )
    assert not (
        analyzer.has_positive_spectral_gap()
    )


def test_single_state_chain_has_zero_spectral_gap():
    analyzer = ErgodicAnalyzer(
        MarkovOperator([
            [1.0],
        ])
    )

    assert np.isclose(
        analyzer.spectral_gap(),
        0.0,
    )


def test_zero_generator_has_zero_spectral_gap():
    analyzer = ErgodicAnalyzer(
        MarkovGenerator([
            [0.0],
        ])
    )

    assert np.isclose(
        analyzer.spectral_gap(),
        0.0,
    )


# ===========================================================================
# Stationary Distributions
# ===========================================================================

def test_discrete_stationary_distribution():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert np.allclose(
        analyzer.stationary_distribution(),
        [0.6, 0.4],
        atol=1e-9,
    )


def test_continuous_stationary_distribution():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    assert np.allclose(
        analyzer.stationary_distribution(),
        [0.25, 0.75],
        atol=1e-9,
    )


def test_stationary_distribution_is_read_only():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    distribution = (
        analyzer.stationary_distribution()
    )

    with pytest.raises(ValueError):
        distribution[0] = 1.0


# ===========================================================================
# Distribution Evolution
# ===========================================================================

def test_discrete_distribution_evolution():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    result = (
        analyzer.evolved_distribution(
            [1.0, 0.0],
            steps=1,
        )
    )

    assert np.allclose(
        result,
        [0.8, 0.2],
    )


def test_continuous_distribution_evolution_at_zero():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    result = (
        analyzer.evolved_distribution(
            [1.0, 0.0],
            time=0.0,
        )
    )

    assert np.allclose(
        result,
        [1.0, 0.0],
    )


def test_discrete_zero_steps_returns_initial_distribution():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    result = (
        analyzer.evolved_distribution(
            [0.25, 0.75],
            steps=0,
        )
    )

    assert np.allclose(
        result,
        [0.25, 0.75],
    )


@pytest.mark.parametrize(
    "steps",
    [
        -1,
        1.5,
        True,
        "10",
    ],
)
def test_discrete_evolution_rejects_invalid_steps(
    steps,
):
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.evolved_distribution(
            [1.0, 0.0],
            steps=steps,
        )


@pytest.mark.parametrize(
    "time",
    [
        -1.0,
        np.inf,
        np.nan,
        True,
        [1.0],
    ],
)
def test_continuous_evolution_rejects_invalid_time(
    time,
):
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    with pytest.raises(OperatorError):
        analyzer.evolved_distribution(
            [1.0, 0.0],
            time=time,
        )


def test_evolution_rejects_dimension_mismatch():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    with pytest.raises(
        DimensionMismatchError
    ):
        analyzer.evolved_distribution(
            [0.2, 0.3, 0.5],
            steps=1,
        )


def test_evolution_rejects_multidimensional_distribution():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.evolved_distribution(
            [[0.6, 0.4]],
            steps=1,
        )


def test_evolution_rejects_negative_distribution():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.evolved_distribution(
            [1.1, -0.1],
            steps=1,
        )


def test_evolution_rejects_unnormalized_distribution():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.evolved_distribution(
            [0.8, 0.8],
            steps=1,
        )


# ===========================================================================
# Total-Variation Distance
# ===========================================================================

def test_total_variation_distance_to_stationarity():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    distance = (
        analyzer.total_variation_distance(
            [1.0, 0.0]
        )
    )

    assert np.isclose(
        distance,
        0.4,
    )


def test_total_variation_distance_with_explicit_reference():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    distance = (
        analyzer.total_variation_distance(
            [0.8, 0.2],
            reference=[0.5, 0.5],
        )
    )

    assert np.isclose(
        distance,
        0.3,
    )


def test_total_variation_distance_is_symmetric_with_reference():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    first = (
        analyzer.total_variation_distance(
            [0.8, 0.2],
            reference=[0.5, 0.5],
        )
    )

    second = (
        analyzer.total_variation_distance(
            [0.5, 0.5],
            reference=[0.8, 0.2],
        )
    )

    assert np.isclose(
        first,
        second,
    )


def test_total_variation_distance_of_equal_distributions_is_zero():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert np.isclose(
        analyzer.total_variation_distance(
            [0.6, 0.4],
            reference=[0.6, 0.4],
        ),
        0.0,
    )


def test_total_variation_rejects_invalid_reference():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.total_variation_distance(
            [0.6, 0.4],
            reference=[0.7, 0.7],
        )


# ===========================================================================
# Convergence Diagnostics
# ===========================================================================

def test_ergodic_chain_convergence_error_is_small():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    error = analyzer.convergence_error(
        [1.0, 0.0],
        steps=100,
    )

    assert error < 1e-12


def test_irreducible_generator_convergence_error_is_small():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    error = analyzer.convergence_error(
        [1.0, 0.0],
        time=100.0,
    )

    assert error < 1e-12


def test_periodic_chain_does_not_converge_ordinary_iterates():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    error = analyzer.convergence_error(
        [1.0, 0.0],
        steps=101,
    )

    assert np.isclose(
        error,
        0.5,
    )


def test_has_converged_for_ergodic_chain():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert analyzer.has_converged(
        [1.0, 0.0],
        steps=100,
        threshold=1e-8,
    )


def test_has_not_converged_for_periodic_chain():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    assert not analyzer.has_converged(
        [1.0, 0.0],
        steps=101,
        threshold=1e-8,
    )


def test_has_converged_accepts_default_threshold():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    assert analyzer.has_converged(
        [1.0, 0.0],
        steps=100,
    )


@pytest.mark.parametrize(
    "threshold",
    [
        -1e-8,
        np.inf,
        np.nan,
        True,
        [1e-8],
    ],
)
def test_has_converged_rejects_invalid_threshold(
    threshold,
):
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.has_converged(
            [1.0, 0.0],
            steps=10,
            threshold=threshold,
        )


# ===========================================================================
# Cesàro Averages
# ===========================================================================

def test_cesaro_average_zero_steps_is_initial_distribution():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    average = analyzer.cesaro_average(
        [1.0, 0.0],
        steps=0,
    )

    assert np.allclose(
        average,
        [1.0, 0.0],
    )


def test_cesaro_average_periodic_chain_even_endpoint():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    average = analyzer.cesaro_average(
        [1.0, 0.0],
        steps=99,
    )

    assert np.allclose(
        average,
        [0.5, 0.5],
        atol=1e-12,
    )


def test_cesaro_average_periodic_chain_converges_to_stationarity():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    average = analyzer.cesaro_average(
        [1.0, 0.0],
        steps=1000,
    )

    assert np.allclose(
        average,
        [501 / 1001, 500 / 1001],
        atol=1e-12,
    )

    assert np.allclose(
        average,
        [0.5, 0.5],
        atol=1e-3,
    )


def test_cesaro_error_periodic_chain_is_small():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    error = analyzer.cesaro_error(
        [1.0, 0.0],
        steps=1000,
    )

    assert error < 1e-3


def test_cesaro_average_ergodic_chain_approaches_stationarity():
    analyzer = ErgodicAnalyzer(
        make_ergodic_operator()
    )

    average = analyzer.cesaro_average(
        [1.0, 0.0],
        steps=1000,
    )

    assert np.allclose(
        average,
        [0.6, 0.4],
        atol=1e-3,
    )


def test_cesaro_average_is_normalized():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    average = analyzer.cesaro_average(
        [1.0, 0.0],
        steps=100,
    )

    assert np.isclose(
        np.sum(average),
        1.0,
    )
    assert np.all(
        average >= 0.0
    )


def test_cesaro_average_is_read_only():
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    average = analyzer.cesaro_average(
        [1.0, 0.0],
        steps=10,
    )

    with pytest.raises(ValueError):
        average[0] = 0.0


@pytest.mark.parametrize(
    "steps",
    [
        -1,
        1.5,
        True,
        "10",
    ],
)
def test_cesaro_average_rejects_invalid_steps(
    steps,
):
    analyzer = ErgodicAnalyzer(
        make_periodic_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.cesaro_average(
            [1.0, 0.0],
            steps=steps,
        )


def test_cesaro_average_rejects_continuous_time_model():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    with pytest.raises(OperatorError):
        analyzer.cesaro_average(
            [1.0, 0.0],
            steps=10,
        )


def test_cesaro_error_rejects_continuous_time_model():
    analyzer = ErgodicAnalyzer(
        make_irreducible_generator()
    )

    with pytest.raises(OperatorError):
        analyzer.cesaro_error(
            [1.0, 0.0],
            steps=10,
        )


# ===========================================================================
# Column-Convention Behavior
# ===========================================================================

def test_column_markov_irreducibility_and_period():
    operator = MarkovOperator(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        convention="column",
    )

    analyzer = ErgodicAnalyzer(
        operator
    )

    assert analyzer.is_irreducible()
    assert analyzer.period() == 2
    assert not analyzer.is_ergodic()


def test_column_markov_ergodicity():
    operator = MarkovOperator(
        [
            [0.8, 0.3],
            [0.2, 0.7],
        ],
        convention="column",
    )

    analyzer = ErgodicAnalyzer(
        operator
    )

    assert analyzer.is_irreducible()
    assert analyzer.period() == 1
    assert analyzer.is_ergodic()


def test_column_generator_ergodicity():
    generator = MarkovGenerator(
        [
            [-0.3, 0.1],
            [0.3, -0.1],
        ],
        convention="column",
    )

    analyzer = ErgodicAnalyzer(
        generator
    )

    assert analyzer.is_irreducible()
    assert analyzer.is_ergodic()
    assert np.isclose(
        analyzer.spectral_gap(),
        0.4,
    )


def test_column_markov_cesaro_average():
    operator = MarkovOperator(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        convention="column",
    )

    analyzer = ErgodicAnalyzer(
        operator
    )

    average = analyzer.cesaro_average(
        [1.0, 0.0],
        steps=99,
    )

    assert np.allclose(
        average,
        [0.5, 0.5],
    )


# ===========================================================================
# Summary
# ===========================================================================

def test_discrete_ergodic_summary():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        states=("a", "b"),
        name="P",
    )

    summary = ErgodicAnalyzer(
        operator
    ).summary()

    assert summary["model"] == "P"
    assert (
        summary["model_type"]
        == "discrete_time"
    )
    assert summary["dimension"] == 2
    assert summary["states"] == ("a", "b")
    assert (
        summary["communicating_classes"]
        == (("a", "b"),)
    )
    assert summary["irreducible"]
    assert summary["period"] == 1
    assert summary["aperiodic"]
    assert summary["ergodic"]
    assert (
        summary["stationary_dimension"]
        == 1
    )
    assert (
        summary[
            "unique_stationary_distribution"
        ]
    )
    assert np.isclose(
        summary["spectral_gap"],
        0.5,
    )
    assert summary[
        "positive_spectral_gap"
    ]


def test_periodic_chain_summary():
    operator = MarkovOperator(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        name="Periodic P",
    )

    summary = ErgodicAnalyzer(
        operator
    ).summary()

    assert (
        summary["model"]
        == "Periodic P"
    )
    assert summary["irreducible"]
    assert summary["period"] == 2
    assert not summary["aperiodic"]
    assert not summary["ergodic"]
    assert np.isclose(
        summary["spectral_gap"],
        0.0,
        atol=1e-12,
    )
    assert not summary[
        "positive_spectral_gap"
    ]


def test_reducible_chain_summary():
    summary = ErgodicAnalyzer(
        make_reducible_operator()
    ).summary()

    assert not summary["irreducible"]
    assert summary["period"] is None
    assert not summary["aperiodic"]
    assert not summary["ergodic"]
    assert (
        summary["stationary_dimension"]
        == 2
    )
    assert not summary[
        "unique_stationary_distribution"
    ]


def test_continuous_time_summary():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
        name="Q",
    )

    summary = ErgodicAnalyzer(
        generator
    ).summary()

    assert summary["model"] == "Q"
    assert (
        summary["model_type"]
        == "continuous_time"
    )
    assert summary["dimension"] == 2
    assert summary["states"] == ("a", "b")
    assert (
        summary["communicating_classes"]
        == (("a", "b"),)
    )
    assert summary["irreducible"]
    assert summary["period"] is None
    assert summary["aperiodic"]
    assert summary["ergodic"]
    assert (
        summary["stationary_dimension"]
        == 1
    )
    assert (
        summary[
            "unique_stationary_distribution"
        ]
    )
    assert np.isclose(
        summary["spectral_gap"],
        0.4,
    )
    assert summary[
        "positive_spectral_gap"
    ]
