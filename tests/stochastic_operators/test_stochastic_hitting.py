"""
Tests for stochastic_operators.hitting.
"""

import numpy as np
import pytest

from operator_core import OperatorError
from stochastic_operators import (
    HittingAnalyzer,
    MarkovGenerator,
    MarkovOperator,
    StochasticOperator,
)


# ===========================================================================
# Shared Test Models
# ===========================================================================

def make_absorbing_operator():
    """
    Return a chain that reaches state 2 almost surely.

    Mean hitting times for target 2 are:

        m_0 = 4,
        m_1 = 2,
        m_2 = 0.
    """

    return MarkovOperator([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ])


def make_partial_hitting_operator():
    """
    Return a chain with competing absorbing states 1 and 2.

    From state 0, target state 1 is reached with probability 1/2.
    """

    return MarkovOperator([
        [0.0, 0.5, 0.5],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


def make_unreachable_operator():
    """
    Return a chain in which state 0 cannot reach target state 2.
    """

    return MarkovOperator([
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ])


def make_generator():
    """
    Return a continuous-time chain with target state 2.

    The rates are:

        0 -> 1 at rate 2,
        1 -> 2 at rate 4.

    Therefore:

        E_0[tau_2] = 1/2 + 1/4 = 3/4,
        E_1[tau_2] = 1/4,
        E_2[tau_2] = 0.
    """

    return MarkovGenerator([
        [-2.0, 2.0, 0.0],
        [0.0, -4.0, 4.0],
        [0.0, 0.0, 0.0],
    ])


def make_partial_hitting_generator():
    """
    Return a generator with competing absorbing states 1 and 2.

    State 0 jumps to either target 1 or absorbing state 2 with equal rates.
    """

    return MarkovGenerator([
        [-2.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])


# ===========================================================================
# Construction and Model Detection
# ===========================================================================

def test_hitting_analyzer_accepts_markov_operator():
    operator = make_absorbing_operator()

    analyzer = HittingAnalyzer(operator)

    assert analyzer.model is operator
    assert analyzer.dimension == 3
    assert analyzer.states == (0, 1, 2)
    assert analyzer.is_discrete_time
    assert not analyzer.is_continuous_time


def test_hitting_analyzer_accepts_markov_generator():
    generator = make_generator()

    analyzer = HittingAnalyzer(generator)

    assert analyzer.model is generator
    assert analyzer.dimension == 3
    assert analyzer.states == (0, 1, 2)
    assert analyzer.is_continuous_time
    assert not analyzer.is_discrete_time


def test_hitting_analyzer_preserves_state_labels():
    operator = MarkovOperator(
        [
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        states=("start", "target"),
    )

    analyzer = HittingAnalyzer(operator)

    assert analyzer.states == (
        "start",
        "target",
    )


def test_hitting_analyzer_rejects_plain_stochastic_operator():
    operator = StochasticOperator([
        [0.5, 0.5],
        [0.0, 1.0],
    ])

    with pytest.raises(OperatorError):
        HittingAnalyzer(operator)


def test_hitting_analyzer_rejects_array():
    with pytest.raises(OperatorError):
        HittingAnalyzer(np.eye(2))


def test_hitting_analyzer_uses_model_tolerance():
    operator = MarkovOperator(
        [
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        tol=1e-8,
    )

    analyzer = HittingAnalyzer(operator)

    assert np.isclose(
        analyzer.tol,
        1e-8,
    )


def test_hitting_analyzer_accepts_custom_tolerance():
    analyzer = HittingAnalyzer(
        make_absorbing_operator(),
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
def test_hitting_analyzer_rejects_invalid_tolerance(
    tol,
):
    with pytest.raises(OperatorError):
        HittingAnalyzer(
            make_absorbing_operator(),
            tol=tol,
        )


# ===========================================================================
# Row-Oriented Matrices
# ===========================================================================

def test_row_oriented_row_markov_matrix():
    matrix = np.array([
        [0.5, 0.5],
        [0.0, 1.0],
    ])

    analyzer = HittingAnalyzer(
        MarkovOperator(matrix)
    )

    assert np.allclose(
        analyzer.row_oriented_matrix(),
        matrix,
    )


def test_row_oriented_column_markov_matrix():
    matrix = np.array([
        [0.5, 0.0],
        [0.5, 1.0],
    ])

    analyzer = HittingAnalyzer(
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
        [-2.0, 2.0],
        [0.0, 0.0],
    ])

    analyzer = HittingAnalyzer(
        MarkovGenerator(matrix)
    )

    assert np.allclose(
        analyzer.row_oriented_matrix(),
        matrix,
    )


def test_row_oriented_column_generator_matrix():
    matrix = np.array([
        [-2.0, 0.0],
        [2.0, 0.0],
    ])

    analyzer = HittingAnalyzer(
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
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    matrix = analyzer.row_oriented_matrix()

    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0


# ===========================================================================
# Adjacency Matrices
# ===========================================================================

def test_discrete_time_adjacency_matrix():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    expected = np.array([
        [True, True, False],
        [False, True, True],
        [False, False, True],
    ])

    assert np.array_equal(
        analyzer.adjacency_matrix(),
        expected,
    )


def test_generator_adjacency_excludes_diagonal():
    analyzer = HittingAnalyzer(
        make_generator()
    )

    expected = np.array([
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ])

    assert np.array_equal(
        analyzer.adjacency_matrix(),
        expected,
    )


def test_adjacency_matrix_respects_analyzer_tolerance():
    operator = MarkovOperator(
        [
            [1.0 - 1e-10, 1e-10],
            [0.0, 1.0],
        ],
        tol=1e-12,
    )

    analyzer = HittingAnalyzer(
        operator,
        tol=1e-8,
    )

    expected = np.array([
        [True, False],
        [False, True],
    ])

    assert np.array_equal(
        analyzer.adjacency_matrix(),
        expected,
    )


def test_adjacency_matrix_is_boolean():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert (
        analyzer.adjacency_matrix().dtype
        == np.dtype(bool)
    )


def test_adjacency_matrix_is_read_only():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    adjacency = analyzer.adjacency_matrix()

    with pytest.raises(ValueError):
        adjacency[0, 0] = False


# ===========================================================================
# State and Target Normalization
# ===========================================================================

def test_state_index_returns_integer_index():
    operator = MarkovOperator(
        [
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        states=("start", "target"),
    )

    analyzer = HittingAnalyzer(operator)

    assert analyzer.state_index(
        "start"
    ) == 0

    assert analyzer.state_index(
        "target"
    ) == 1


def test_state_index_rejects_unknown_state():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.state_index("missing")


def test_target_indices_accepts_single_state():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert analyzer.target_indices(
        2
    ) == (2,)


def test_target_indices_accepts_multiple_states():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert analyzer.target_indices(
        [2, 1]
    ) == (1, 2)


def test_target_indices_removes_duplicates():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert analyzer.target_indices(
        [2, 1, 2, 1]
    ) == (1, 2)


def test_target_indices_accepts_generator():
    analyzer = HittingAnalyzer(
        make_generator()
    )

    assert analyzer.target_indices(
        (1, 2)
    ) == (1, 2)


def test_target_indices_rejects_empty_iterable():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.target_indices([])


def test_target_indices_rejects_unknown_target():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.target_indices([2, 9])


def test_target_indices_rejects_unknown_string():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    with pytest.raises(OperatorError):
        analyzer.target_indices("missing")


def test_target_states_returns_normalized_labels():
    operator = MarkovOperator(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        states=("a", "b", "c"),
    )

    analyzer = HittingAnalyzer(operator)

    assert analyzer.target_states(
        ["c", "b", "c"]
    ) == ("b", "c")


def test_target_mask_for_single_target():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    expected = np.array([
        False,
        False,
        True,
    ])

    assert np.array_equal(
        analyzer.target_mask(2),
        expected,
    )


def test_target_mask_for_multiple_targets():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    expected = np.array([
        False,
        True,
        True,
    ])

    assert np.array_equal(
        analyzer.target_mask([1, 2]),
        expected,
    )


def test_target_mask_is_read_only():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    mask = analyzer.target_mask(2)

    with pytest.raises(ValueError):
        mask[0] = True


# ===========================================================================
# Reachability
# ===========================================================================

def test_all_states_can_reach_absorbing_target():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    expected = np.array([
        True,
        True,
        True,
    ])

    assert np.array_equal(
        analyzer.reachable_mask(2),
        expected,
    )


def test_unreachable_state_is_excluded():
    analyzer = HittingAnalyzer(
        make_unreachable_operator()
    )

    expected = np.array([
        False,
        True,
        True,
    ])

    assert np.array_equal(
        analyzer.reachable_mask(2),
        expected,
    )


def test_generator_reachability():
    analyzer = HittingAnalyzer(
        make_generator()
    )

    expected = np.array([
        True,
        True,
        True,
    ])

    assert np.array_equal(
        analyzer.reachable_mask(2),
        expected,
    )


def test_reachable_states_returns_labels():
    operator = MarkovOperator(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        states=("isolated", "middle", "target"),
    )

    analyzer = HittingAnalyzer(operator)

    assert analyzer.reachable_states(
        "target"
    ) == (
        "middle",
        "target",
    )


def test_can_reach_returns_true():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert analyzer.can_reach(
        0,
        2,
    )


def test_can_reach_returns_false():
    analyzer = HittingAnalyzer(
        make_unreachable_operator()
    )

    assert not analyzer.can_reach(
        0,
        2,
    )


def test_target_always_reaches_itself():
    analyzer = HittingAnalyzer(
        make_unreachable_operator()
    )

    assert analyzer.can_reach(
        2,
        2,
    )


def test_reachable_mask_is_read_only():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    mask = analyzer.reachable_mask(2)

    with pytest.raises(ValueError):
        mask[0] = False


# ===========================================================================
# Discrete-Time Hitting Probabilities
# ===========================================================================

def test_discrete_almost_sure_hitting_probabilities():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    probabilities = (
        analyzer.hitting_probabilities(2)
    )

    assert np.allclose(
        probabilities,
        [1.0, 1.0, 1.0],
    )


def test_discrete_partial_hitting_probabilities():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    probabilities = (
        analyzer.hitting_probabilities(1)
    )

    assert np.allclose(
        probabilities,
        [0.5, 1.0, 0.0],
    )


def test_discrete_unreachable_target_probability_is_zero():
    analyzer = HittingAnalyzer(
        make_unreachable_operator()
    )

    probabilities = (
        analyzer.hitting_probabilities(2)
    )

    assert np.allclose(
        probabilities,
        [0.0, 1.0, 1.0],
    )


def test_discrete_multiple_target_probabilities():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    probabilities = (
        analyzer.hitting_probabilities(
            [1, 2]
        )
    )

    assert np.allclose(
        probabilities,
        [1.0, 1.0, 1.0],
    )


def test_target_state_hitting_probability_is_one():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    assert np.isclose(
        analyzer.hitting_probability(
            1,
            1,
        ),
        1.0,
    )


def test_state_specific_partial_hitting_probability():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    assert np.isclose(
        analyzer.hitting_probability(
            0,
            1,
        ),
        0.5,
    )


def test_hitting_probabilities_are_read_only():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    probabilities = (
        analyzer.hitting_probabilities(2)
    )

    with pytest.raises(ValueError):
        probabilities[0] = 0.0


# ===========================================================================
# Continuous-Time Hitting Probabilities
# ===========================================================================

def test_generator_almost_sure_hitting_probabilities():
    analyzer = HittingAnalyzer(
        make_generator()
    )

    probabilities = (
        analyzer.hitting_probabilities(2)
    )

    assert np.allclose(
        probabilities,
        [1.0, 1.0, 1.0],
    )


def test_generator_partial_hitting_probabilities():
    analyzer = HittingAnalyzer(
        make_partial_hitting_generator()
    )

    probabilities = (
        analyzer.hitting_probabilities(1)
    )

    assert np.allclose(
        probabilities,
        [0.5, 1.0, 0.0],
    )


def test_generator_multiple_target_probabilities():
    analyzer = HittingAnalyzer(
        make_partial_hitting_generator()
    )

    probabilities = (
        analyzer.hitting_probabilities(
            [1, 2]
        )
    )

    assert np.allclose(
        probabilities,
        [1.0, 1.0, 1.0],
    )


# ===========================================================================
# Almost-Sure Hitting
# ===========================================================================

def test_almost_sure_mask_for_absorbing_chain():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    expected = np.array([
        True,
        True,
        True,
    ])

    assert np.array_equal(
        analyzer.almost_sure_mask(2),
        expected,
    )


def test_almost_sure_mask_for_partial_hitting_chain():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    expected = np.array([
        False,
        True,
        False,
    ])

    assert np.array_equal(
        analyzer.almost_sure_mask(1),
        expected,
    )


def test_almost_sure_states_return_labels():
    operator = MarkovOperator(
        [
            [0.0, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        states=("start", "left", "right"),
    )

    analyzer = HittingAnalyzer(operator)

    assert analyzer.almost_sure_states(
        "left"
    ) == ("left",)


def test_is_almost_surely_hitting_true():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert analyzer.is_almost_surely_hitting(
        0,
        2,
    )


def test_is_almost_surely_hitting_false():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    assert not analyzer.is_almost_surely_hitting(
        0,
        1,
    )


def test_almost_sure_mask_is_read_only():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    mask = analyzer.almost_sure_mask(2)

    with pytest.raises(ValueError):
        mask[0] = False


# ===========================================================================
# Discrete-Time Mean Hitting Times
# ===========================================================================

def test_discrete_mean_hitting_times():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    times = analyzer.mean_hitting_times(
        2
    )

    assert np.allclose(
        times,
        [4.0, 2.0, 0.0],
    )


def test_discrete_state_specific_mean_hitting_time():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert np.isclose(
        analyzer.mean_hitting_time(
            0,
            2,
        ),
        4.0,
    )


def test_target_mean_hitting_time_is_zero():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    assert np.isclose(
        analyzer.mean_hitting_time(
            2,
            2,
        ),
        0.0,
    )


def test_partial_hitting_has_infinite_unconditional_mean():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    times = analyzer.mean_hitting_times(
        1
    )

    assert np.isinf(times[0])
    assert np.isclose(
        times[1],
        0.0,
    )
    assert np.isinf(times[2])


def test_unreachable_target_has_infinite_mean():
    analyzer = HittingAnalyzer(
        make_unreachable_operator()
    )

    times = analyzer.mean_hitting_times(
        2
    )

    assert np.isinf(times[0])
    assert np.isclose(
        times[1],
        2.0,
    )
    assert np.isclose(
        times[2],
        0.0,
    )


def test_multiple_targets_have_zero_mean_on_target_set():
    analyzer = HittingAnalyzer(
        make_partial_hitting_operator()
    )

    times = analyzer.mean_hitting_times(
        [1, 2]
    )

    assert np.allclose(
        times,
        [1.0, 0.0, 0.0],
    )


def test_mean_hitting_times_are_read_only():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    times = analyzer.mean_hitting_times(
        2
    )

    with pytest.raises(ValueError):
        times[0] = 0.0


# ===========================================================================
# Continuous-Time Mean Hitting Times
# ===========================================================================

def test_generator_mean_hitting_times():
    analyzer = HittingAnalyzer(
        make_generator()
    )

    times = analyzer.mean_hitting_times(
        2
    )

    assert np.allclose(
        times,
        [0.75, 0.25, 0.0],
    )


def test_generator_state_specific_mean_hitting_time():
    analyzer = HittingAnalyzer(
        make_generator()
    )

    assert np.isclose(
        analyzer.mean_hitting_time(
            0,
            2,
        ),
        0.75,
    )


def test_generator_partial_hitting_has_infinite_mean():
    analyzer = HittingAnalyzer(
        make_partial_hitting_generator()
    )

    times = analyzer.mean_hitting_times(
        1
    )

    assert np.isinf(times[0])
    assert np.isclose(
        times[1],
        0.0,
    )
    assert np.isinf(times[2])


def test_generator_multiple_targets_mean_hitting_time():
    analyzer = HittingAnalyzer(
        make_partial_hitting_generator()
    )

    times = analyzer.mean_hitting_times(
        [1, 2]
    )

    assert np.allclose(
        times,
        [0.5, 0.0, 0.0],
    )


# ===========================================================================
# Column-Convention Behavior
# ===========================================================================

def test_column_markov_hitting_probabilities():
    matrix = np.array([
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 1.0],
    ])

    analyzer = HittingAnalyzer(
        MarkovOperator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.hitting_probabilities(2),
        [1.0, 1.0, 1.0],
    )


def test_column_markov_mean_hitting_times():
    matrix = np.array([
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 1.0],
    ])

    analyzer = HittingAnalyzer(
        MarkovOperator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.mean_hitting_times(2),
        [4.0, 2.0, 0.0],
    )


def test_column_generator_hitting_probabilities():
    matrix = np.array([
        [-2.0, 0.0, 0.0],
        [2.0, -4.0, 0.0],
        [0.0, 4.0, 0.0],
    ])

    analyzer = HittingAnalyzer(
        MarkovGenerator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.hitting_probabilities(2),
        [1.0, 1.0, 1.0],
    )


def test_column_generator_mean_hitting_times():
    matrix = np.array([
        [-2.0, 0.0, 0.0],
        [2.0, -4.0, 0.0],
        [0.0, 4.0, 0.0],
    ])

    analyzer = HittingAnalyzer(
        MarkovGenerator(
            matrix,
            convention="column",
        )
    )

    assert np.allclose(
        analyzer.mean_hitting_times(2),
        [0.75, 0.25, 0.0],
    )


# ===========================================================================
# Target-Set Edge Cases
# ===========================================================================

def test_every_state_as_target():
    analyzer = HittingAnalyzer(
        make_absorbing_operator()
    )

    probabilities = (
        analyzer.hitting_probabilities(
            [0, 1, 2]
        )
    )

    times = analyzer.mean_hitting_times(
        [0, 1, 2]
    )

    assert np.allclose(
        probabilities,
        [1.0, 1.0, 1.0],
    )

    assert np.allclose(
        times,
        [0.0, 0.0, 0.0],
    )


def test_single_state_chain_target():
    analyzer = HittingAnalyzer(
        MarkovOperator([
            [1.0],
        ])
    )

    assert np.allclose(
        analyzer.hitting_probabilities(0),
        [1.0],
    )

    assert np.allclose(
        analyzer.mean_hitting_times(0),
        [0.0],
    )


def test_zero_generator_single_state_target():
    analyzer = HittingAnalyzer(
        MarkovGenerator([
            [0.0],
        ])
    )

    assert np.allclose(
        analyzer.hitting_probabilities(0),
        [1.0],
    )

    assert np.allclose(
        analyzer.mean_hitting_times(0),
        [0.0],
    )


# ===========================================================================
# Summary
# ===========================================================================

def test_discrete_hitting_summary():
    operator = MarkovOperator(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        states=("a", "b", "c"),
        name="P",
    )

    summary = HittingAnalyzer(
        operator
    ).summary(
        "c",
        start="a",
    )

    assert summary["model"] == "P"
    assert (
        summary["model_type"]
        == "discrete_time"
    )
    assert summary["dimension"] == 3
    assert summary["states"] == (
        "a",
        "b",
        "c",
    )
    assert summary["targets"] == (
        "c",
    )
    assert summary[
        "reachable_states"
    ] == (
        "a",
        "b",
        "c",
    )
    assert summary[
        "almost_sure_states"
    ] == (
        "a",
        "b",
        "c",
    )
    assert np.allclose(
        summary[
            "hitting_probabilities"
        ],
        [1.0, 1.0, 1.0],
    )
    assert np.allclose(
        summary[
            "mean_hitting_times"
        ],
        [4.0, 2.0, 0.0],
    )
    assert summary["start"] == "a"
    assert summary["start_can_reach"]
    assert np.isclose(
        summary[
            "start_hitting_probability"
        ],
        1.0,
    )
    assert summary[
        "start_almost_sure"
    ]
    assert np.isclose(
        summary[
            "start_mean_hitting_time"
        ],
        4.0,
    )


def test_partial_hitting_summary():
    operator = MarkovOperator(
        [
            [0.0, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        states=("start", "target", "failure"),
        name="Partial P",
    )

    summary = HittingAnalyzer(
        operator
    ).summary(
        "target",
        start="start",
    )

    assert (
        summary["model"]
        == "Partial P"
    )
    assert summary["targets"] == (
        "target",
    )
    assert summary[
        "reachable_states"
    ] == (
        "start",
        "target",
    )
    assert summary[
        "almost_sure_states"
    ] == (
        "target",
    )
    assert np.allclose(
        summary[
            "hitting_probabilities"
        ],
        [0.5, 1.0, 0.0],
    )
    assert np.isinf(
        summary[
            "mean_hitting_times"
        ][0]
    )
    assert summary[
        "start_can_reach"
    ]
    assert np.isclose(
        summary[
            "start_hitting_probability"
        ],
        0.5,
    )
    assert not summary[
        "start_almost_sure"
    ]
    assert np.isinf(
        summary[
            "start_mean_hitting_time"
        ]
    )


def test_continuous_time_hitting_summary():
    generator = MarkovGenerator(
        [
            [-2.0, 2.0, 0.0],
            [0.0, -4.0, 4.0],
            [0.0, 0.0, 0.0],
        ],
        states=("a", "b", "c"),
        name="Q",
    )

    summary = HittingAnalyzer(
        generator
    ).summary(
        "c",
        start="a",
    )

    assert summary["model"] == "Q"
    assert (
        summary["model_type"]
        == "continuous_time"
    )
    assert summary["targets"] == (
        "c",
    )
    assert np.allclose(
        summary[
            "hitting_probabilities"
        ],
        [1.0, 1.0, 1.0],
    )
    assert np.allclose(
        summary[
            "mean_hitting_times"
        ],
        [0.75, 0.25, 0.0],
    )
    assert np.isclose(
        summary[
            "start_mean_hitting_time"
        ],
        0.75,
    )


def test_summary_without_start_omits_start_fields():
    summary = HittingAnalyzer(
        make_absorbing_operator()
    ).summary(2)

    assert "start" not in summary
    assert (
        "start_can_reach"
        not in summary
    )
    assert (
        "start_hitting_probability"
        not in summary
    )
    assert (
        "start_almost_sure"
        not in summary
    )
    assert (
        "start_mean_hitting_time"
        not in summary
    )
