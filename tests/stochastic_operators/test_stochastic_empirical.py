"""
Tests for stochastic_operators.monte_carlo.empirical.

This suite covers the empirical-estimation layer:

- scalar validation helpers;
- state and path validation;
- path extraction and state inference;
- burn-in and thinning;
- numerical helper functions;
- empirical state distributions;
- empirical transition matrices;
- empirical stationary distributions;
- empirical hitting probabilities;
- empirical hitting times;
- empirical return times;
- immutable MonteCarloResult integration.

The tests use exact finite paths wherever possible. No probabilistic
tolerances or asymptotic claims are required.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from stochastic_operators.monte_carlo import (
    MonteCarloResult,
    empirical_distribution,
    empirical_hitting_probability,
    empirical_hitting_time,
    empirical_return_time,
    empirical_stationary_distribution,
    empirical_transition_matrix,
    simulate_chain,
    simulate_paths,
)
from stochastic_operators.monte_carlo.empirical import (
    _extract_paths,
    _first_hitting_time,
    _flatten_paths,
    _freeze_state_path,
    _freeze_states,
    _infer_states,
    _mean_or_nan,
    _readonly_float_array,
    _readonly_integer_array,
    _resolve_states,
    _sample_variance,
    _standard_error,
    _state_index,
    _trim_paths,
    _validate_boolean,
    _validate_nonnegative_integer,
    _validate_positive_integer,
    _validate_target,
    _validate_zero_row_policy,
)


# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def simple_path() -> tuple[str, ...]:
    return ("a", "a", "b", "a", "b")


@pytest.fixture
def transition_path() -> tuple[str, ...]:
    return ("a", "b", "a", "a")


@pytest.fixture
def two_path_result() -> MonteCarloResult:
    return MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b", "a"),
            ("b", "b", "a"),
        ),
        states=("a", "b"),
        steps=2,
        n_paths=2,
        seed=42,
    )


@pytest.fixture
def three_state_paths() -> MonteCarloResult:
    return MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b", "c"),
            ("a", "a", "c"),
            ("b", "a", "b"),
        ),
        states=("a", "b", "c"),
        steps=2,
        n_paths=3,
    )


def assert_read_only(array: np.ndarray) -> None:
    """Assert that an array cannot be mutated."""

    assert isinstance(array, np.ndarray)
    assert array.flags.writeable is False

    if array.size:
        with pytest.raises(ValueError):
            array.flat[0] = 100


# ============================================================================
# Integer validation helpers
# ============================================================================


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        10,
        np.int32(4),
        np.int64(8),
    ],
)
def test_validate_nonnegative_integer_accepts_valid_values(value) -> None:
    result = _validate_nonnegative_integer(
        value,
        name="burn_in",
    )

    assert result == int(value)
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        1.5,
        "1",
        None,
        object(),
    ],
)
def test_validate_nonnegative_integer_rejects_invalid_types(value) -> None:
    with pytest.raises(TypeError, match="nonnegative integer"):
        _validate_nonnegative_integer(
            value,
            name="burn_in",
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -10,
        np.int64(-5),
    ],
)
def test_validate_nonnegative_integer_rejects_negative_values(value) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_nonnegative_integer(
            value,
            name="burn_in",
        )


@pytest.mark.parametrize(
    "value",
    [
        1,
        2,
        10,
        np.int64(5),
    ],
)
def test_validate_positive_integer_accepts_valid_values(value) -> None:
    result = _validate_positive_integer(
        value,
        name="thinning",
    )

    assert result == int(value)
    assert isinstance(result, int)


def test_validate_positive_integer_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        _validate_positive_integer(
            0,
            name="thinning",
        )


def test_validate_positive_integer_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_positive_integer(
            -1,
            name="thinning",
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        1.5,
        "2",
        None,
    ],
)
def test_validate_positive_integer_rejects_invalid_type(value) -> None:
    with pytest.raises(TypeError, match="nonnegative integer"):
        _validate_positive_integer(
            value,
            name="thinning",
        )


# ============================================================================
# Boolean validation
# ============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.bool_(False), False),
    ],
)
def test_validate_boolean_accepts_boolean_values(
    value,
    expected: bool,
) -> None:
    result = _validate_boolean(
        value,
        name="include_initial",
    )

    assert result is expected
    assert isinstance(result, bool)


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        "true",
        None,
        object(),
    ],
)
def test_validate_boolean_rejects_non_boolean_values(value) -> None:
    with pytest.raises(TypeError, match="Boolean"):
        _validate_boolean(
            value,
            name="include_initial",
        )


# ============================================================================
# Zero-row policy validation
# ============================================================================


@pytest.mark.parametrize(
    "policy",
    [
        "nan",
        "zeros",
        "self",
    ],
)
def test_validate_zero_row_policy_accepts_valid_policies(
    policy: str,
) -> None:
    assert _validate_zero_row_policy(policy) == policy


@pytest.mark.parametrize(
    "policy",
    [
        "",
        "zero",
        "identity",
        "NAN",
        "SELF",
    ],
)
def test_validate_zero_row_policy_rejects_unknown_policy(
    policy: str,
) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        _validate_zero_row_policy(policy)


@pytest.mark.parametrize(
    "policy",
    [
        None,
        1,
        True,
        object(),
    ],
)
def test_validate_zero_row_policy_rejects_non_string(policy) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        _validate_zero_row_policy(policy)


# ============================================================================
# State-path freezing
# ============================================================================


def test_freeze_state_path_converts_sequence_to_tuple() -> None:
    result = _freeze_state_path(
        ["a", "b", "c"],
        name="path",
    )

    assert result == ("a", "b", "c")
    assert isinstance(result, tuple)


def test_freeze_state_path_preserves_tuple_valued_states() -> None:
    path = (
        ("x", 1),
        ("y", 2),
    )

    result = _freeze_state_path(path)

    assert result == path


def test_freeze_state_path_accepts_mixed_hashable_states() -> None:
    result = _freeze_state_path(
        [
            "a",
            1,
            ("b", 2),
        ]
    )

    assert result == (
        "a",
        1,
        ("b", 2),
    )


def test_freeze_state_path_rejects_empty_path_by_default() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _freeze_state_path([])


def test_freeze_state_path_allows_empty_path_when_requested() -> None:
    assert (
        _freeze_state_path(
            [],
            allow_empty=True,
        )
        == ()
    )


@pytest.mark.parametrize(
    "path",
    [
        "abc",
        b"abc",
    ],
)
def test_freeze_state_path_rejects_string_like_path(path) -> None:
    with pytest.raises(TypeError, match="not a string"):
        _freeze_state_path(path)


@pytest.mark.parametrize(
    "path",
    [
        None,
        42,
        object(),
    ],
)
def test_freeze_state_path_rejects_non_iterable(path) -> None:
    with pytest.raises(TypeError, match="finite sequence"):
        _freeze_state_path(path)


def test_freeze_state_path_rejects_unhashable_state() -> None:
    with pytest.raises(TypeError, match="hashable"):
        _freeze_state_path(
            (
                "a",
                ["b"],
            )
        )


# ============================================================================
# State-space freezing
# ============================================================================


def test_freeze_states_returns_ordered_tuple() -> None:
    result = _freeze_states(["c", "a", "b"])

    assert result == ("c", "a", "b")


def test_freeze_states_accepts_single_state() -> None:
    assert _freeze_states(["only"]) == ("only",)


def test_freeze_states_rejects_empty_state_space() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _freeze_states([])


def test_freeze_states_rejects_duplicate_states() -> None:
    with pytest.raises(ValueError, match="unique"):
        _freeze_states(
            [
                "a",
                "b",
                "a",
            ]
        )


def test_freeze_states_detects_equal_cross_type_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        _freeze_states(
            [
                1,
                True,
            ]
        )


def test_freeze_states_rejects_unhashable_state() -> None:
    with pytest.raises(TypeError, match="hashable"):
        _freeze_states(
            [
                "a",
                ["b"],
            ]
        )


# ============================================================================
# Path extraction
# ============================================================================


def test_extract_paths_from_raw_path() -> None:
    result = _extract_paths(
        ("a", "b", "c")
    )

    assert result == (
        ("a", "b", "c"),
    )


def test_extract_paths_treats_tuple_valued_states_as_one_path() -> None:
    raw_path = (
        ("a", 1),
        ("b", 2),
    )

    result = _extract_paths(raw_path)

    assert result == (raw_path,)


def test_extract_paths_from_single_path_result() -> None:
    source = MonteCarloResult(
        method="simulate_chain",
        path=("a", "b", "a"),
        states=("a", "b"),
        steps=2,
        n_paths=1,
    )

    result = _extract_paths(source)

    assert result == (
        ("a", "b", "a"),
    )


def test_extract_paths_from_multiple_path_result(
    two_path_result: MonteCarloResult,
) -> None:
    result = _extract_paths(two_path_result)

    assert result == (
        ("a", "b", "a"),
        ("b", "b", "a"),
    )


def test_extract_paths_rejects_result_without_paths() -> None:
    source = MonteCarloResult(
        method="estimate",
        estimate=0.5,
    )

    with pytest.raises(
        ValueError,
        match="neither path nor paths",
    ):
        _extract_paths(source)


def test_extract_paths_rejects_empty_paths_collection() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(),
        n_paths=0,
    )

    with pytest.raises(
        ValueError,
        match="at least one path",
    ):
        _extract_paths(source)


def test_extract_paths_rejects_empty_raw_path() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _extract_paths(())


# ============================================================================
# State inference and resolution
# ============================================================================


def test_infer_states_uses_first_observation_order() -> None:
    paths = (
        ("c", "a", "b"),
        ("b", "d", "a"),
    )

    result = _infer_states(paths)

    assert result == (
        "c",
        "a",
        "b",
        "d",
    )


def test_infer_states_deduplicates_repeated_observations() -> None:
    result = _infer_states(
        (
            ("a", "a", "b"),
            ("b", "a", "b"),
        )
    )

    assert result == ("a", "b")


def test_infer_states_supports_tuple_labels() -> None:
    result = _infer_states(
        (
            (
                ("x", 1),
                ("y", 2),
                ("x", 1),
            ),
        )
    )

    assert result == (
        ("x", 1),
        ("y", 2),
    )


def test_infer_states_rejects_no_observations() -> None:
    with pytest.raises(ValueError, match="empty paths"):
        _infer_states(((), ()))


def test_resolve_states_infers_states_when_none() -> None:
    result = _resolve_states(
        (
            ("b", "a", "c"),
        ),
        states=None,
    )

    assert result == (
        "b",
        "a",
        "c",
    )


def test_resolve_states_preserves_explicit_order() -> None:
    result = _resolve_states(
        (
            ("a", "b", "a"),
        ),
        states=("b", "a", "c"),
    )

    assert result == (
        "b",
        "a",
        "c",
    )


def test_resolve_states_allows_unobserved_explicit_states() -> None:
    result = _resolve_states(
        (
            ("a", "a"),
        ),
        states=("a", "b", "c"),
    )

    assert result == (
        "a",
        "b",
        "c",
    )


def test_resolve_states_rejects_observation_outside_state_space() -> None:
    with pytest.raises(
        ValueError,
        match="not present in states",
    ):
        _resolve_states(
            (
                ("a", "c"),
            ),
            states=("a", "b"),
        )


def test_resolve_states_error_identifies_path_and_position() -> None:
    with pytest.raises(
        ValueError,
        match=r"path 1, position 1",
    ):
        _resolve_states(
            (
                ("a", "b"),
                ("a", "c"),
            ),
            states=("a", "b"),
        )


# ============================================================================
# Burn-in and thinning
# ============================================================================


def test_trim_paths_without_processing_returns_original_values() -> None:
    paths = (
        ("a", "b", "c"),
        ("c", "b", "a"),
    )

    result = _trim_paths(
        paths,
        burn_in=0,
        thinning=1,
    )

    assert result == paths


def test_trim_paths_applies_burn_in_to_each_path() -> None:
    result = _trim_paths(
        (
            ("a", "b", "c", "d"),
            ("w", "x", "y", "z"),
        ),
        burn_in=2,
        thinning=1,
    )

    assert result == (
        ("c", "d"),
        ("y", "z"),
    )


def test_trim_paths_applies_thinning_to_each_path() -> None:
    result = _trim_paths(
        (
            ("a", "b", "c", "d", "e"),
            ("v", "w", "x", "y", "z"),
        ),
        burn_in=0,
        thinning=2,
    )

    assert result == (
        ("a", "c", "e"),
        ("v", "x", "z"),
    )


def test_trim_paths_applies_burn_in_before_thinning() -> None:
    result = _trim_paths(
        (
            ("a", "b", "c", "d", "e", "f"),
        ),
        burn_in=1,
        thinning=2,
    )

    assert result == (
        ("b", "d", "f"),
    )


def test_trim_paths_enforces_minimum_length() -> None:
    with pytest.raises(
        ValueError,
        match="fewer than 2 retained observations",
    ):
        _trim_paths(
            (
                ("a", "b"),
            ),
            burn_in=1,
            thinning=1,
            minimum_length=2,
        )


def test_trim_paths_rejects_burn_in_removing_all_observations() -> None:
    with pytest.raises(
        ValueError,
        match="fewer than 1 retained observations",
    ):
        _trim_paths(
            (
                ("a", "b"),
            ),
            burn_in=2,
            thinning=1,
        )


@pytest.mark.parametrize(
    "burn_in",
    [
        -1,
        True,
        1.5,
    ],
)
def test_trim_paths_rejects_invalid_burn_in(burn_in) -> None:
    with pytest.raises((TypeError, ValueError)):
        _trim_paths(
            (("a", "b"),),
            burn_in=burn_in,
            thinning=1,
        )


@pytest.mark.parametrize(
    "thinning",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_trim_paths_rejects_invalid_thinning(thinning) -> None:
    with pytest.raises((TypeError, ValueError)):
        _trim_paths(
            (("a", "b"),),
            burn_in=0,
            thinning=thinning,
        )


def test_trim_paths_rejects_zero_minimum_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        _trim_paths(
            (("a", "b"),),
            burn_in=0,
            thinning=1,
            minimum_length=0,
        )


# ============================================================================
# Flattening and state indexing
# ============================================================================


def test_flatten_paths_preserves_path_order() -> None:
    result = _flatten_paths(
        (
            ("a", "b"),
            ("c", "d"),
        )
    )

    assert result == (
        "a",
        "b",
        "c",
        "d",
    )


def test_flatten_paths_handles_empty_paths() -> None:
    assert _flatten_paths(((), ())) == ()


def test_flatten_paths_does_not_insert_boundary_markers() -> None:
    result = _flatten_paths(
        (
            ("a",),
            ("b",),
        )
    )

    assert result == ("a", "b")


def test_state_index_maps_states_to_ordered_indices() -> None:
    result = _state_index(
        ("c", "a", "b")
    )

    assert result == {
        "c": 0,
        "a": 1,
        "b": 2,
    }


def test_state_index_supports_tuple_labels() -> None:
    result = _state_index(
        (
            ("x", 1),
            ("y", 2),
        )
    )

    assert result == {
        ("x", 1): 0,
        ("y", 2): 1,
    }


# ============================================================================
# Read-only array helpers
# ============================================================================


def test_readonly_float_array_returns_float64_array() -> None:
    result = _readonly_float_array([1, 2, 3])

    assert result.dtype == np.float64
    np.testing.assert_array_equal(
        result,
        np.array([1.0, 2.0, 3.0]),
    )


def test_readonly_float_array_returns_independent_copy() -> None:
    source = np.array([1.0, 2.0])

    result = _readonly_float_array(source)
    source[0] = 100.0

    np.testing.assert_array_equal(
        result,
        np.array([1.0, 2.0]),
    )


def test_readonly_float_array_is_read_only() -> None:
    result = _readonly_float_array([1.0, 2.0])

    assert_read_only(result)


def test_readonly_float_array_preserves_shape() -> None:
    result = _readonly_float_array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    assert result.shape == (2, 2)


def test_readonly_integer_array_returns_int64_array() -> None:
    result = _readonly_integer_array([1, 2, 3])

    assert result.dtype == np.int64
    np.testing.assert_array_equal(
        result,
        np.array([1, 2, 3]),
    )


def test_readonly_integer_array_returns_independent_copy() -> None:
    source = np.array([1, 2], dtype=np.int64)

    result = _readonly_integer_array(source)
    source[0] = 100

    np.testing.assert_array_equal(
        result,
        np.array([1, 2]),
    )


def test_readonly_integer_array_is_read_only() -> None:
    result = _readonly_integer_array([1, 2])

    assert_read_only(result)


# ============================================================================
# Statistical helper functions
# ============================================================================


@pytest.mark.parametrize(
    "samples",
    [
        np.array([], dtype=float),
        np.array([1.0]),
    ],
)
def test_sample_variance_returns_none_for_fewer_than_two_samples(
    samples: np.ndarray,
) -> None:
    assert _sample_variance(samples) is None


def test_sample_variance_uses_unbiased_estimator() -> None:
    samples = np.array(
        [1.0, 2.0, 3.0],
        dtype=float,
    )

    result = _sample_variance(samples)

    assert result == pytest.approx(1.0)


def test_sample_variance_of_constant_samples_is_zero() -> None:
    samples = np.array(
        [4.0, 4.0, 4.0],
        dtype=float,
    )

    assert _sample_variance(samples) == pytest.approx(0.0)


def test_standard_error_returns_none_without_variance() -> None:
    assert (
        _standard_error(
            None,
            n_samples=10,
        )
        is None
    )


def test_standard_error_returns_none_for_zero_samples() -> None:
    assert (
        _standard_error(
            1.0,
            n_samples=0,
        )
        is None
    )


def test_standard_error_is_square_root_variance_over_n() -> None:
    result = _standard_error(
        4.0,
        n_samples=16,
    )

    assert result == pytest.approx(0.5)


def test_standard_error_of_zero_variance_is_zero() -> None:
    assert (
        _standard_error(
            0.0,
            n_samples=10,
        )
        == pytest.approx(0.0)
    )


def test_mean_or_nan_returns_mean_for_nonempty_samples() -> None:
    result = _mean_or_nan(
        np.array([1.0, 2.0, 6.0])
    )

    assert result == pytest.approx(3.0)


def test_mean_or_nan_returns_nan_for_empty_samples() -> None:
    result = _mean_or_nan(
        np.array([], dtype=float)
    )

    assert np.isnan(result)


# ============================================================================
# Target validation
# ============================================================================


def test_validate_target_accepts_known_target() -> None:
    result = _validate_target(
        "b",
        states=("a", "b"),
    )

    assert result == "b"


def test_validate_target_supports_custom_name() -> None:
    result = _validate_target(
        "a",
        states=("a", "b"),
        name="state",
    )

    assert result == "a"


def test_validate_target_rejects_unknown_target() -> None:
    with pytest.raises(
        ValueError,
        match="not present in states",
    ):
        _validate_target(
            "c",
            states=("a", "b"),
        )


def test_validate_target_rejects_unhashable_target() -> None:
    with pytest.raises(TypeError, match="hashable"):
        _validate_target(
            ["a"],
            states=("a", "b"),
        )


# ============================================================================
# First-hitting-time helper
# ============================================================================


def test_first_hitting_time_includes_initial_state() -> None:
    result = _first_hitting_time(
        ("a", "b", "c"),
        target="a",
        include_initial=True,
    )

    assert result == 0


def test_first_hitting_time_can_exclude_initial_state() -> None:
    result = _first_hitting_time(
        ("a", "b", "a"),
        target="a",
        include_initial=False,
    )

    assert result == 2


def test_first_hitting_time_returns_first_visit() -> None:
    result = _first_hitting_time(
        ("a", "b", "c", "b"),
        target="b",
        include_initial=True,
    )

    assert result == 1


def test_first_hitting_time_returns_none_when_target_absent() -> None:
    result = _first_hitting_time(
        ("a", "b", "a"),
        target="c",
        include_initial=True,
    )

    assert result is None


def test_first_hitting_time_excluding_initial_can_return_none() -> None:
    result = _first_hitting_time(
        ("a", "b", "c"),
        target="a",
        include_initial=False,
    )

    assert result is None


# ============================================================================
# empirical_distribution
# ============================================================================


def test_empirical_distribution_returns_result(
    simple_path: tuple[str, ...],
) -> None:
    result = empirical_distribution(simple_path)

    assert isinstance(result, MonteCarloResult)


def test_empirical_distribution_sets_method(
    simple_path: tuple[str, ...],
) -> None:
    result = empirical_distribution(simple_path)

    assert result.method == "empirical_distribution"


def test_empirical_distribution_computes_exact_frequencies(
    simple_path: tuple[str, ...],
) -> None:
    result = empirical_distribution(
        simple_path,
        states=("a", "b"),
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array([0.6, 0.4]),
    )


def test_empirical_distribution_infers_first_observation_order() -> None:
    result = empirical_distribution(
        ("b", "a", "b", "c"),
    )

    assert result.states == (
        "b",
        "a",
        "c",
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array([0.5, 0.25, 0.25]),
    )


def test_empirical_distribution_respects_explicit_state_order() -> None:
    result = empirical_distribution(
        ("a", "a", "b"),
        states=("b", "a"),
    )

    assert result.states == ("b", "a")

    np.testing.assert_allclose(
        result.estimate,
        np.array([1.0 / 3.0, 2.0 / 3.0]),
    )


def test_empirical_distribution_includes_unobserved_explicit_states() -> None:
    result = empirical_distribution(
        ("a", "a", "b"),
        states=("a", "b", "c"),
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array([2.0 / 3.0, 1.0 / 3.0, 0.0]),
    )

    assert result.metadata["counts"] == (
        2,
        1,
        0,
    )


def test_empirical_distribution_estimate_is_read_only() -> None:
    result = empirical_distribution(
        ("a", "b", "a"),
    )

    assert_read_only(result.estimate)


def test_empirical_distribution_records_sample_count() -> None:
    result = empirical_distribution(
        ("a", "b", "a", "b"),
    )

    assert result.n_samples == 4


def test_empirical_distribution_records_single_path_count() -> None:
    result = empirical_distribution(
        ("a", "b", "a"),
    )

    assert result.n_paths == 1


def test_empirical_distribution_records_metadata_counts() -> None:
    result = empirical_distribution(
        ("a", "a", "b", "c", "a"),
        states=("a", "b", "c"),
    )

    assert result.metadata["counts"] == (
        3,
        1,
        1,
    )


def test_empirical_distribution_records_relative_frequency_estimator() -> None:
    result = empirical_distribution(
        ("a", "b"),
    )

    assert result.metadata["estimator"] == "relative_frequency"


def test_empirical_distribution_marks_single_path_as_not_pooled() -> None:
    result = empirical_distribution(
        ("a", "b"),
    )

    assert result.metadata["pooled_paths"] is False


def test_empirical_distribution_pools_multiple_paths(
    two_path_result: MonteCarloResult,
) -> None:
    result = empirical_distribution(
        two_path_result,
        states=("a", "b"),
    )

    # Paths contain a total of three a's and three b's.
    np.testing.assert_allclose(
        result.estimate,
        np.array([0.5, 0.5]),
    )

    assert result.n_samples == 6
    assert result.n_paths == 2
    assert result.metadata["pooled_paths"] is True


def test_empirical_distribution_applies_burn_in_per_path(
    two_path_result: MonteCarloResult,
) -> None:
    result = empirical_distribution(
        two_path_result,
        states=("a", "b"),
        burn_in=1,
    )

    # Retained paths: (b, a), (b, a)
    np.testing.assert_allclose(
        result.estimate,
        np.array([0.5, 0.5]),
    )

    assert result.n_samples == 4
    assert result.burn_in == 1


def test_empirical_distribution_applies_thinning_per_path() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b", "a", "b", "a"),
            ("b", "a", "b", "a", "b"),
        ),
        steps=4,
        n_paths=2,
    )

    result = empirical_distribution(
        source,
        states=("a", "b"),
        thinning=2,
    )

    # Retained paths: (a, a, a), (b, b, b)
    np.testing.assert_allclose(
        result.estimate,
        np.array([0.5, 0.5]),
    )

    assert result.n_samples == 6
    assert result.thinning == 2


def test_empirical_distribution_accepts_simulate_chain_result() -> None:
    simulation = simulate_chain(
        [[0.0, 1.0], [1.0, 0.0]],
        states=("a", "b"),
        initial_state="a",
        steps=4,
        seed=42,
    )

    result = empirical_distribution(simulation)

    np.testing.assert_allclose(
        result.estimate,
        np.array([0.6, 0.4]),
    )


def test_empirical_distribution_rejects_unknown_observation() -> None:
    with pytest.raises(
        ValueError,
        match="not present in states",
    ):
        empirical_distribution(
            ("a", "c"),
            states=("a", "b"),
        )


@pytest.mark.parametrize(
    "burn_in",
    [
        -1,
        True,
        1.5,
    ],
)
def test_empirical_distribution_rejects_invalid_burn_in(
    burn_in,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        empirical_distribution(
            ("a", "b"),
            burn_in=burn_in,
        )


@pytest.mark.parametrize(
    "thinning",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_empirical_distribution_rejects_invalid_thinning(
    thinning,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        empirical_distribution(
            ("a", "b"),
            thinning=thinning,
        )


def test_empirical_distribution_rejects_complete_burn_in() -> None:
    with pytest.raises(
        ValueError,
        match="fewer than 1 retained observations",
    ):
        empirical_distribution(
            ("a", "b"),
            burn_in=2,
        )


# ============================================================================
# empirical_transition_matrix
# ============================================================================


def test_empirical_transition_matrix_returns_result(
    transition_path: tuple[str, ...],
) -> None:
    result = empirical_transition_matrix(
        transition_path,
        states=("a", "b"),
    )

    assert isinstance(result, MonteCarloResult)


def test_empirical_transition_matrix_sets_method(
    transition_path: tuple[str, ...],
) -> None:
    result = empirical_transition_matrix(
        transition_path,
        states=("a", "b"),
    )

    assert result.method == "empirical_transition_matrix"


def test_empirical_transition_matrix_computes_exact_matrix(
    transition_path: tuple[str, ...],
) -> None:
    result = empirical_transition_matrix(
        transition_path,
        states=("a", "b"),
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array(
            [
                [0.5, 0.5],
                [1.0, 0.0],
            ]
        ),
    )


def test_empirical_transition_matrix_records_transition_count(
    transition_path: tuple[str, ...],
) -> None:
    result = empirical_transition_matrix(
        transition_path,
        states=("a", "b"),
    )

    assert result.n_samples == 3


def test_empirical_transition_matrix_records_count_matrix(
    transition_path: tuple[str, ...],
) -> None:
    result = empirical_transition_matrix(
        transition_path,
        states=("a", "b"),
    )

    assert result.metadata["transition_counts"] == (
        (1, 1),
        (1, 0),
    )


def test_empirical_transition_matrix_records_row_totals(
    transition_path: tuple[str, ...],
) -> None:
    result = empirical_transition_matrix(
        transition_path,
        states=("a", "b"),
    )

    assert result.metadata["row_totals"] == (
        2,
        1,
    )


def test_empirical_transition_matrix_estimate_is_read_only(
    transition_path: tuple[str, ...],
) -> None:
    result = empirical_transition_matrix(
        transition_path,
        states=("a", "b"),
    )

    assert_read_only(result.estimate)


def test_empirical_transition_matrix_does_not_create_path_boundary_transition(
    two_path_result: MonteCarloResult,
) -> None:
    result = empirical_transition_matrix(
        two_path_result,
        states=("a", "b"),
    )

    # First path: a->b, b->a
    # Second path: b->b, b->a
    # There must be no artificial a->b transition between the paths.
    assert result.metadata["transition_counts"] == (
        (0, 1),
        (2, 1),
    )

    assert result.n_samples == 4


def test_empirical_transition_matrix_pools_multiple_paths(
    two_path_result: MonteCarloResult,
) -> None:
    result = empirical_transition_matrix(
        two_path_result,
        states=("a", "b"),
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array(
            [
                [0.0, 1.0],
                [2.0 / 3.0, 1.0 / 3.0],
            ]
        ),
    )

    assert result.n_paths == 2
    assert result.metadata["pooled_paths"] is True


def test_empirical_transition_matrix_applies_burn_in() -> None:
    result = empirical_transition_matrix(
        ("a", "b", "a", "a", "b"),
        states=("a", "b"),
        burn_in=1,
    )

    # Retained: b, a, a, b
    np.testing.assert_allclose(
        result.estimate,
        np.array(
            [
                [0.5, 0.5],
                [1.0, 0.0],
            ]
        ),
    )


def test_empirical_transition_matrix_applies_thinning() -> None:
    result = empirical_transition_matrix(
        ("a", "b", "a", "b", "a"),
        states=("a", "b"),
        thinning=2,
        zero_row="zeros",
    )

    # Retained: a, a, a
    np.testing.assert_allclose(
        result.estimate,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )

    assert result.n_samples == 2


def test_empirical_transition_matrix_nan_policy() -> None:
    result = empirical_transition_matrix(
        ("a", "a"),
        states=("a", "b"),
        zero_row="nan",
    )

    np.testing.assert_allclose(
        result.estimate[0],
        np.array([1.0, 0.0]),
    )

    assert np.all(np.isnan(result.estimate[1]))
    assert result.metadata["unobserved_rows"] == ("b",)


def test_empirical_transition_matrix_zero_policy() -> None:
    result = empirical_transition_matrix(
        ("a", "a"),
        states=("a", "b"),
        zero_row="zeros",
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_empirical_transition_matrix_self_policy() -> None:
    result = empirical_transition_matrix(
        ("a", "a"),
        states=("a", "b"),
        zero_row="self",
    )

    np.testing.assert_allclose(
        result.estimate,
        np.eye(2),
    )


def test_empirical_transition_matrix_one_observation_nan_policy() -> None:
    result = empirical_transition_matrix(
        ("a",),
        states=("a", "b"),
        zero_row="nan",
    )

    assert result.n_samples == 0
    assert np.all(np.isnan(result.estimate))
    assert result.metadata["unobserved_rows"] == (
        "a",
        "b",
    )


def test_empirical_transition_matrix_one_observation_self_policy() -> None:
    result = empirical_transition_matrix(
        ("a",),
        states=("a", "b"),
        zero_row="self",
    )

    assert result.n_samples == 0

    np.testing.assert_allclose(
        result.estimate,
        np.eye(2),
    )


def test_empirical_transition_matrix_infers_state_order() -> None:
    result = empirical_transition_matrix(
        ("b", "a", "b"),
    )

    assert result.states == ("b", "a")

    np.testing.assert_allclose(
        result.estimate,
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )


def test_empirical_transition_matrix_records_zero_row_policy() -> None:
    result = empirical_transition_matrix(
        ("a",),
        states=("a", "b"),
        zero_row="self",
    )

    assert result.metadata["zero_row"] == "self"


@pytest.mark.parametrize(
    "policy",
    [
        "",
        "invalid",
        None,
        1,
    ],
)
def test_empirical_transition_matrix_rejects_invalid_zero_row_policy(
    policy,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        empirical_transition_matrix(
            ("a", "b"),
            zero_row=policy,
        )


def test_empirical_transition_matrix_rejects_complete_burn_in() -> None:
    with pytest.raises(
        ValueError,
        match="fewer than 1 retained observations",
    ):
        empirical_transition_matrix(
            ("a", "b"),
            burn_in=2,
        )


# ============================================================================
# empirical_stationary_distribution
# ============================================================================


def test_empirical_stationary_distribution_returns_result() -> None:
    result = empirical_stationary_distribution(
        ("a", "b", "a"),
    )

    assert isinstance(result, MonteCarloResult)


def test_empirical_stationary_distribution_sets_method() -> None:
    result = empirical_stationary_distribution(
        ("a", "b", "a"),
    )

    assert result.method == "empirical_stationary_distribution"


def test_empirical_stationary_distribution_matches_occupation_measure() -> None:
    result = empirical_stationary_distribution(
        ("a", "a", "b", "a", "b"),
        states=("a", "b"),
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array([0.6, 0.4]),
    )


def test_empirical_stationary_distribution_preserves_distribution_counts() -> None:
    result = empirical_stationary_distribution(
        ("a", "a", "b"),
        states=("a", "b"),
    )

    assert result.metadata["counts"] == (
        2,
        1,
    )


def test_empirical_stationary_distribution_records_estimator() -> None:
    result = empirical_stationary_distribution(
        ("a", "b"),
    )

    assert result.metadata["estimator"] == "occupation_measure"


def test_empirical_stationary_distribution_records_assumption() -> None:
    result = empirical_stationary_distribution(
        ("a", "b"),
    )

    assert result.metadata["stationarity_assumed"] is True
    assert result.metadata["stationarity_verified"] is False


def test_empirical_stationary_distribution_preserves_processing_fields() -> None:
    result = empirical_stationary_distribution(
        ("a", "b", "a", "b", "a"),
        burn_in=1,
        thinning=2,
    )

    assert result.burn_in == 1
    assert result.thinning == 2
    assert result.n_samples == 2


def test_empirical_stationary_distribution_estimate_is_read_only() -> None:
    result = empirical_stationary_distribution(
        ("a", "b", "a"),
    )

    assert_read_only(result.estimate)


# ============================================================================
# empirical_hitting_probability
# ============================================================================


def test_empirical_hitting_probability_returns_result() -> None:
    result = empirical_hitting_probability(
        ("a", "b"),
        target="b",
    )

    assert isinstance(result, MonteCarloResult)


def test_empirical_hitting_probability_sets_method() -> None:
    result = empirical_hitting_probability(
        ("a", "b"),
        target="b",
    )

    assert result.method == "empirical_hitting_probability"


def test_empirical_hitting_probability_single_path_hit() -> None:
    result = empirical_hitting_probability(
        ("a", "b", "c"),
        target="c",
    )

    assert result.estimate == pytest.approx(1.0)
    np.testing.assert_array_equal(
        result.samples,
        np.array([1.0]),
    )


def test_empirical_hitting_probability_single_path_miss_with_explicit_target() -> None:
    result = empirical_hitting_probability(
        ("a", "b"),
        target="c",
        states=("a", "b", "c"),
    )

    assert result.estimate == pytest.approx(0.0)
    np.testing.assert_array_equal(
        result.samples,
        np.array([0.0]),
    )


def test_empirical_hitting_probability_multiple_paths() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b"),
            ("a", "a"),
            ("b", "b"),
            ("a", "c"),
        ),
        states=("a", "b", "c"),
        steps=1,
        n_paths=4,
    )

    result = empirical_hitting_probability(
        source,
        target="b",
    )

    # Paths 0 and 2 hit b.
    assert result.estimate == pytest.approx(0.5)

    np.testing.assert_array_equal(
        result.samples,
        np.array([1.0, 0.0, 1.0, 0.0]),
    )

    assert result.metadata["hits"] == 2
    assert result.metadata["misses"] == 2


def test_empirical_hitting_probability_computes_sample_variance() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b"),
            ("a", "a"),
        ),
        states=("a", "b"),
        steps=1,
        n_paths=2,
    )

    result = empirical_hitting_probability(
        source,
        target="b",
    )

    # Unbiased variance of [1, 0].
    assert result.variance == pytest.approx(0.5)
    assert result.standard_error == pytest.approx(0.5)


def test_empirical_hitting_probability_one_path_has_no_variance() -> None:
    result = empirical_hitting_probability(
        ("a", "b"),
        target="b",
    )

    assert result.variance is None
    assert result.standard_error is None


def test_empirical_hitting_probability_includes_initial_by_default() -> None:
    result = empirical_hitting_probability(
        ("a", "b"),
        target="a",
    )

    assert result.estimate == pytest.approx(1.0)
    assert result.metadata["include_initial"] is True


def test_empirical_hitting_probability_can_exclude_initial() -> None:
    result = empirical_hitting_probability(
        ("a", "b"),
        target="a",
        include_initial=False,
    )

    assert result.estimate == pytest.approx(0.0)
    assert result.metadata["include_initial"] is False


def test_empirical_hitting_probability_excluding_initial_counts_return() -> None:
    result = empirical_hitting_probability(
        ("a", "b", "a"),
        target="a",
        include_initial=False,
    )

    assert result.estimate == pytest.approx(1.0)


def test_empirical_hitting_probability_records_path_count() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b"),
            ("a", "a"),
            ("b", "a"),
        ),
        states=("a", "b"),
        steps=1,
        n_paths=3,
    )

    result = empirical_hitting_probability(
        source,
        target="b",
    )

    assert result.n_samples == 3
    assert result.n_paths == 3


def test_empirical_hitting_probability_samples_are_read_only() -> None:
    result = empirical_hitting_probability(
        ("a", "b"),
        target="b",
    )

    assert_read_only(result.samples)


def test_empirical_hitting_probability_rejects_unknown_target() -> None:
    with pytest.raises(
        ValueError,
        match="not present in states",
    ):
        empirical_hitting_probability(
            ("a", "b"),
            target="c",
        )


@pytest.mark.parametrize(
    "include_initial",
    [
        0,
        1,
        "yes",
        None,
    ],
)
def test_empirical_hitting_probability_rejects_invalid_boolean(
    include_initial,
) -> None:
    with pytest.raises(TypeError, match="Boolean"):
        empirical_hitting_probability(
            ("a", "b"),
            target="b",
            include_initial=include_initial,
        )


# ============================================================================
# empirical_hitting_time
# ============================================================================


def test_empirical_hitting_time_returns_result() -> None:
    result = empirical_hitting_time(
        ("a", "b"),
        target="b",
    )

    assert isinstance(result, MonteCarloResult)


def test_empirical_hitting_time_sets_method() -> None:
    result = empirical_hitting_time(
        ("a", "b"),
        target="b",
    )

    assert result.method == "empirical_hitting_time"


def test_empirical_hitting_time_computes_single_first_hit() -> None:
    result = empirical_hitting_time(
        ("a", "a", "b", "c"),
        target="c",
    )

    assert result.estimate == pytest.approx(3.0)

    np.testing.assert_array_equal(
        result.samples,
        np.array([3.0]),
    )


def test_empirical_hitting_time_includes_initial_state() -> None:
    result = empirical_hitting_time(
        ("a", "b"),
        target="a",
    )

    assert result.estimate == pytest.approx(0.0)
    assert result.metadata["include_initial"] is True


def test_empirical_hitting_time_can_exclude_initial_state() -> None:
    result = empirical_hitting_time(
        ("a", "b", "a"),
        target="a",
        include_initial=False,
    )

    assert result.estimate == pytest.approx(2.0)


def test_empirical_hitting_time_multiple_paths() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b", "c"),
            ("a", "c", "c"),
            ("c", "a", "a"),
        ),
        states=("a", "b", "c"),
        steps=2,
        n_paths=3,
    )

    result = empirical_hitting_time(
        source,
        target="c",
    )

    np.testing.assert_array_equal(
        result.samples,
        np.array([2.0, 1.0, 0.0]),
    )

    assert result.estimate == pytest.approx(1.0)
    assert result.n_samples == 3


def test_empirical_hitting_time_uses_only_successful_paths() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b", "c"),
            ("a", "b", "a"),
            ("c", "a", "a"),
        ),
        states=("a", "b", "c"),
        steps=2,
        n_paths=3,
    )

    result = empirical_hitting_time(
        source,
        target="c",
    )

    np.testing.assert_array_equal(
        result.samples,
        np.array([2.0, 0.0]),
    )

    assert result.estimate == pytest.approx(1.0)
    assert result.n_samples == 2
    assert result.n_paths == 3
    assert result.metadata["successful_paths"] == 2
    assert result.metadata["censored_paths"] == 1


def test_empirical_hitting_time_all_paths_censored() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b"),
            ("b", "a"),
        ),
        states=("a", "b", "c"),
        steps=1,
        n_paths=2,
    )

    result = empirical_hitting_time(
        source,
        target="c",
    )

    assert np.isnan(result.estimate)
    assert result.samples.size == 0
    assert result.n_samples == 0
    assert result.variance is None
    assert result.standard_error is None
    assert result.metadata["successful_paths"] == 0
    assert result.metadata["censored_paths"] == 2


def test_empirical_hitting_time_computes_variance_and_standard_error() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "c", "c"),
            ("a", "b", "c"),
            ("c", "a", "a"),
        ),
        states=("a", "b", "c"),
        steps=2,
        n_paths=3,
    )

    result = empirical_hitting_time(
        source,
        target="c",
    )

    # Samples are [1, 2, 0].
    assert result.variance == pytest.approx(1.0)
    assert result.standard_error == pytest.approx(
        np.sqrt(1.0 / 3.0)
    )


def test_empirical_hitting_time_records_omit_censoring_policy() -> None:
    result = empirical_hitting_time(
        ("a", "b"),
        target="c",
        states=("a", "b", "c"),
    )

    assert result.metadata["censoring_policy"] == "omit"


def test_empirical_hitting_time_samples_are_read_only() -> None:
    result = empirical_hitting_time(
        ("a", "b", "c"),
        target="c",
    )

    assert_read_only(result.samples)


def test_empirical_hitting_time_rejects_unknown_target() -> None:
    with pytest.raises(
        ValueError,
        match="not present in states",
    ):
        empirical_hitting_time(
            ("a", "b"),
            target="c",
        )


# ============================================================================
# empirical_return_time
# ============================================================================


def test_empirical_return_time_returns_result() -> None:
    result = empirical_return_time(
        ("a", "b", "a"),
        state="a",
    )

    assert isinstance(result, MonteCarloResult)


def test_empirical_return_time_sets_method() -> None:
    result = empirical_return_time(
        ("a", "b", "a"),
        state="a",
    )

    assert result.method == "empirical_return_time"


def test_empirical_return_time_computes_intervisit_times() -> None:
    result = empirical_return_time(
        ("a", "b", "a", "a", "b", "a"),
        state="a",
    )

    np.testing.assert_array_equal(
        result.samples,
        np.array([2.0, 1.0, 2.0]),
    )

    assert result.estimate == pytest.approx(5.0 / 3.0)


def test_empirical_return_time_consecutive_visits_have_time_one() -> None:
    result = empirical_return_time(
        ("a", "a", "a"),
        state="a",
    )

    np.testing.assert_array_equal(
        result.samples,
        np.array([1.0, 1.0]),
    )

    assert result.estimate == pytest.approx(1.0)


def test_empirical_return_time_single_visit_has_no_return() -> None:
    result = empirical_return_time(
        ("a", "b", "c"),
        state="a",
    )

    assert np.isnan(result.estimate)
    assert result.samples.size == 0
    assert result.n_samples == 0
    assert result.variance is None
    assert result.standard_error is None


def test_empirical_return_time_absent_explicit_state_has_no_return() -> None:
    result = empirical_return_time(
        ("a", "b", "a"),
        state="c",
        states=("a", "b", "c"),
    )

    assert np.isnan(result.estimate)
    assert result.samples.size == 0


def test_empirical_return_time_collects_returns_within_each_path() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b", "a"),
            ("a", "a", "b", "a"),
        ),
        states=("a", "b"),
        n_paths=2,
    )

    result = empirical_return_time(
        source,
        state="a",
    )

    # First path contributes 2.
    # Second path contributes 1 and 2.
    np.testing.assert_array_equal(
        result.samples,
        np.array([2.0, 1.0, 2.0]),
    )


def test_empirical_return_time_does_not_cross_path_boundaries() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b"),
            ("b", "a"),
        ),
        states=("a", "b"),
        n_paths=2,
    )

    result = empirical_return_time(
        source,
        state="a",
    )

    # Each path contains exactly one visit. Combining the paths would
    # incorrectly create a return interval, but independent paths must not.
    assert result.samples.size == 0
    assert np.isnan(result.estimate)


def test_empirical_return_time_records_paths_with_returns() -> None:
    source = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b", "a"),
            ("a", "b", "b"),
            ("b", "a", "a"),
        ),
        states=("a", "b"),
        n_paths=3,
    )

    result = empirical_return_time(
        source,
        state="a",
    )

    assert result.metadata["paths_with_returns"] == 2
    assert result.metadata["paths_without_returns"] == 1


def test_empirical_return_time_computes_variance() -> None:
    result = empirical_return_time(
        ("a", "b", "a", "b", "b", "a"),
        state="a",
    )

    # Return times are [2, 3].
    assert result.estimate == pytest.approx(2.5)
    assert result.variance == pytest.approx(0.5)
    assert result.standard_error == pytest.approx(0.5)


def test_empirical_return_time_one_return_has_no_variance() -> None:
    result = empirical_return_time(
        ("a", "b", "a"),
        state="a",
    )

    assert result.samples.tolist() == [2.0]
    assert result.variance is None
    assert result.standard_error is None


def test_empirical_return_time_samples_are_read_only() -> None:
    result = empirical_return_time(
        ("a", "b", "a"),
        state="a",
    )

    assert_read_only(result.samples)


def test_empirical_return_time_rejects_unknown_state() -> None:
    with pytest.raises(
        ValueError,
        match="not present in states",
    ):
        empirical_return_time(
            ("a", "b"),
            state="c",
        )


# ============================================================================
# Result immutability and metadata
# ============================================================================


@pytest.mark.parametrize(
    "estimator",
    [
        lambda: empirical_distribution(
            ("a", "b", "a"),
        ),
        lambda: empirical_transition_matrix(
            ("a", "b", "a"),
        ),
        lambda: empirical_stationary_distribution(
            ("a", "b", "a"),
        ),
        lambda: empirical_hitting_probability(
            ("a", "b", "a"),
            target="b",
        ),
        lambda: empirical_hitting_time(
            ("a", "b", "a"),
            target="b",
        ),
        lambda: empirical_return_time(
            ("a", "b", "a"),
            state="a",
        ),
    ],
)
def test_empirical_results_have_read_only_metadata(estimator) -> None:
    result = estimator()

    assert isinstance(result.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


@pytest.mark.parametrize(
    "estimator",
    [
        lambda: empirical_distribution(
            ("a", "b", "a"),
        ),
        lambda: empirical_transition_matrix(
            ("a", "b", "a"),
        ),
        lambda: empirical_stationary_distribution(
            ("a", "b", "a"),
        ),
        lambda: empirical_hitting_probability(
            ("a", "b", "a"),
            target="b",
        ),
        lambda: empirical_hitting_time(
            ("a", "b", "a"),
            target="b",
        ),
        lambda: empirical_return_time(
            ("a", "b", "a"),
            state="a",
        ),
    ],
)
def test_empirical_results_store_immutable_state_tuple(estimator) -> None:
    result = estimator()

    assert isinstance(result.states, tuple)


# ============================================================================
# Public API integration
# ============================================================================


@pytest.mark.parametrize(
    "function",
    [
        empirical_distribution,
        empirical_transition_matrix,
        empirical_stationary_distribution,
        empirical_hitting_probability,
        empirical_hitting_time,
        empirical_return_time,
    ],
)
def test_empirical_public_functions_are_callable(function) -> None:
    assert callable(function)


def test_simulated_alternating_chain_has_exact_empirical_distribution() -> None:
    simulation = simulate_chain(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        states=("a", "b"),
        initial_state="a",
        steps=4,
        seed=42,
    )

    result = empirical_distribution(simulation)

    np.testing.assert_allclose(
        result.estimate,
        np.array([0.6, 0.4]),
    )


def test_simulated_alternating_chain_has_exact_transition_matrix() -> None:
    simulation = simulate_chain(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        states=("a", "b"),
        initial_state="a",
        steps=10,
        seed=42,
    )

    result = empirical_transition_matrix(simulation)

    np.testing.assert_allclose(
        result.estimate,
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )


def test_simulated_identity_paths_have_exact_hitting_probability() -> None:
    simulation = simulate_paths(
        np.eye(2),
        states=("a", "b"),
        initial_state="a",
        steps=5,
        n_paths=4,
        seed=42,
    )

    result = empirical_hitting_probability(
        simulation,
        target="b",
    )

    assert result.estimate == pytest.approx(0.0)
    assert result.metadata["hits"] == 0
    assert result.metadata["misses"] == 4


def test_simulated_alternating_paths_have_exact_hitting_time() -> None:
    simulation = simulate_paths(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        states=("a", "b"),
        initial_state="a",
        steps=4,
        n_paths=3,
        seed=42,
    )

    result = empirical_hitting_time(
        simulation,
        target="b",
    )

    np.testing.assert_array_equal(
        result.samples,
        np.array([1.0, 1.0, 1.0]),
    )

    assert result.estimate == pytest.approx(1.0)
    assert result.variance == pytest.approx(0.0)
    assert result.standard_error == pytest.approx(0.0)


def test_simulated_alternating_paths_have_exact_return_times() -> None:
    simulation = simulate_paths(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        states=("a", "b"),
        initial_state="a",
        steps=4,
        n_paths=2,
        seed=42,
    )

    result = empirical_return_time(
        simulation,
        state="a",
    )

    # Each path is a, b, a, b, a and contributes [2, 2].
    np.testing.assert_array_equal(
        result.samples,
        np.array([2.0, 2.0, 2.0, 2.0]),
    )

    assert result.estimate == pytest.approx(2.0)


def test_distribution_and_stationary_estimators_agree_numerically() -> None:
    path = (
        "a",
        "a",
        "b",
        "c",
        "a",
        "b",
    )

    distribution = empirical_distribution(
        path,
        states=("a", "b", "c"),
        burn_in=1,
        thinning=2,
    )

    stationary = empirical_stationary_distribution(
        path,
        states=("a", "b", "c"),
        burn_in=1,
        thinning=2,
    )

    np.testing.assert_allclose(
        distribution.estimate,
        stationary.estimate,
    )

    assert distribution.states == stationary.states
    assert distribution.n_samples == stationary.n_samples
