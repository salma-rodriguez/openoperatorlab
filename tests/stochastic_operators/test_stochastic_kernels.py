"""
Tests for stochastic_operators.kernels.

The suite verifies:

- finite stochastic-kernel construction;
- row and column conventions;
- validation and normalization;
- immutable matrices and metadata;
- state-label handling;
- transition probabilities and supports;
- factory constructors;
- convention conversion and relabeling;
- composition and powers;
- MarkovOperator conversion;
- summaries and representations.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from stochastic_operators.kernels import (
    Kernel,
    StochasticKernel,
)
from stochastic_operators.markov import MarkovOperator


# ============================================================================
# Fixtures and helpers
# ============================================================================


@pytest.fixture
def row_matrix() -> np.ndarray:
    """A simple three-state row-stochastic matrix."""

    return np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.25, 0.75],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


@pytest.fixture
def column_matrix(row_matrix: np.ndarray) -> np.ndarray:
    """Column-oriented representation of ``row_matrix``."""

    return row_matrix.T.copy()


@pytest.fixture
def labeled_kernel(row_matrix: np.ndarray) -> StochasticKernel:
    """A labeled row-oriented stochastic kernel."""

    return StochasticKernel(
        row_matrix,
        states=("a", "b", "c"),
        name="K",
        metadata={"purpose": "testing"},
    )


def assert_read_only(array: np.ndarray) -> None:
    """Assert that a NumPy array rejects mutation."""

    assert array.flags.writeable is False

    with pytest.raises(ValueError):
        array.flat[0] = 123.0


# ============================================================================
# Public aliases
# ============================================================================


def test_kernel_is_alias_for_stochastic_kernel() -> None:
    assert Kernel is StochasticKernel


# ============================================================================
# Basic construction
# ============================================================================


def test_construct_row_kernel(row_matrix: np.ndarray) -> None:
    kernel = StochasticKernel(row_matrix)

    assert kernel.shape == (3, 3)
    assert kernel.size == 3
    assert len(kernel) == 3
    assert kernel.states == (0, 1, 2)
    assert kernel.convention == "row"
    assert kernel.name == "StochasticKernel"
    assert kernel.tol == pytest.approx(1e-10)
    assert kernel.is_square is True
    assert kernel.is_finite is True

    np.testing.assert_allclose(kernel.matrix, row_matrix)


def test_construct_column_kernel(column_matrix: np.ndarray) -> None:
    kernel = StochasticKernel(
        column_matrix,
        convention="column",
    )

    assert kernel.convention == "column"
    np.testing.assert_allclose(kernel.matrix, column_matrix)


def test_constructor_preserves_custom_state_labels(
    row_matrix: np.ndarray,
) -> None:
    kernel = StochasticKernel(
        row_matrix,
        states=("red", "green", "blue"),
    )

    assert kernel.states == ("red", "green", "blue")
    assert kernel.state_to_index == {
        "red": 0,
        "green": 1,
        "blue": 2,
    }


def test_constructor_preserves_name_and_metadata(
    row_matrix: np.ndarray,
) -> None:
    kernel = StochasticKernel(
        row_matrix,
        name="weather",
        metadata={"units": "probability"},
    )

    assert kernel.name == "weather"
    assert kernel.metadata == {"units": "probability"}
    assert isinstance(kernel.metadata, MappingProxyType)


def test_constructor_copies_input_matrix(
    row_matrix: np.ndarray,
) -> None:
    original = row_matrix.copy()
    kernel = StochasticKernel(row_matrix)

    row_matrix[0, 0] = 0.0

    np.testing.assert_allclose(kernel.matrix, original)


def test_matrix_is_read_only(
    labeled_kernel: StochasticKernel,
) -> None:
    assert_read_only(labeled_kernel.matrix)


def test_state_mapping_is_read_only(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(TypeError):
        labeled_kernel.state_to_index["d"] = 3


def test_metadata_is_read_only(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(TypeError):
        labeled_kernel.metadata["new"] = "value"


# ============================================================================
# Constructor validation
# ============================================================================


@pytest.mark.parametrize(
    "matrix",
    [
        [0.5, 0.5],
        np.array([0.5, 0.5]),
        np.zeros((2, 2, 2)),
    ],
)
def test_rejects_non_two_dimensional_matrix(matrix) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        StochasticKernel(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.empty((0, 0)),
        np.empty((0, 2)),
        np.empty((2, 0)),
    ],
)
def test_rejects_empty_matrix(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        StochasticKernel(matrix)


def test_rejects_nonsquare_matrix() -> None:
    matrix = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )

    with pytest.raises(ValueError, match="square"):
        StochasticKernel(matrix)


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_nonfinite_entries(bad_value: float) -> None:
    matrix = np.eye(2)
    matrix[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        StochasticKernel(matrix)


def test_rejects_materially_negative_probability() -> None:
    matrix = np.array(
        [
            [1.1, -0.1],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="negative"):
        StochasticKernel(matrix)


def test_accepts_tiny_negative_roundoff_within_tolerance() -> None:
    matrix = np.array(
        [
            [1.0 + 1e-12, -1e-12],
            [0.0, 1.0],
        ]
    )

    kernel = StochasticKernel(matrix, tol=1e-10)

    np.testing.assert_allclose(kernel.matrix, matrix)


def test_rejects_row_sums_not_equal_to_one() -> None:
    matrix = np.array(
        [
            [0.2, 0.2],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(ValueError, match="sum to one"):
        StochasticKernel(matrix, convention="row")


def test_rejects_column_sums_not_equal_to_one() -> None:
    matrix = np.array(
        [
            [0.2, 0.5],
            [0.2, 0.5],
        ]
    )

    with pytest.raises(ValueError, match="sum to one"):
        StochasticKernel(matrix, convention="column")


@pytest.mark.parametrize(
    "convention",
    [
        "rows",
        "columns",
        "left",
        "",
        None,
    ],
)
def test_rejects_invalid_convention(convention) -> None:
    with pytest.raises(ValueError, match="convention"):
        StochasticKernel(
            np.eye(2),
            convention=convention,
        )


@pytest.mark.parametrize(
    "tol",
    [
        True,
        False,
        "small",
        object(),
    ],
)
def test_rejects_nonreal_tolerance(tol) -> None:
    with pytest.raises(TypeError, match="tol"):
        StochasticKernel(np.eye(2), tol=tol)


@pytest.mark.parametrize(
    "tol",
    [
        -1.0,
        -1e-12,
        np.inf,
        -np.inf,
        np.nan,
    ],
)
def test_rejects_invalid_numeric_tolerance(tol: float) -> None:
    with pytest.raises(ValueError, match="tol"):
        StochasticKernel(np.eye(2), tol=tol)


def test_rejects_wrong_number_of_state_labels() -> None:
    with pytest.raises(ValueError, match="number of state labels"):
        StochasticKernel(
            np.eye(3),
            states=("a", "b"),
        )


def test_rejects_duplicate_state_labels() -> None:
    with pytest.raises(ValueError, match="unique"):
        StochasticKernel(
            np.eye(3),
            states=("a", "a", "b"),
        )


def test_rejects_unhashable_state_label() -> None:
    with pytest.raises(TypeError, match="hashable"):
        StochasticKernel(
            np.eye(2),
            states=(["a"], ["b"]),
        )


# ============================================================================
# Constructor normalization
# ============================================================================


def test_constructor_can_normalize_row_weights() -> None:
    weights = np.array(
        [
            [2.0, 2.0],
            [1.0, 3.0],
        ]
    )

    kernel = StochasticKernel(
        weights,
        normalize=True,
    )

    expected = np.array(
        [
            [0.5, 0.5],
            [0.25, 0.75],
        ]
    )

    np.testing.assert_allclose(kernel.matrix, expected)


def test_constructor_can_normalize_column_weights() -> None:
    weights = np.array(
        [
            [2.0, 1.0],
            [2.0, 3.0],
        ]
    )

    kernel = StochasticKernel(
        weights,
        convention="column",
        normalize=True,
    )

    expected = np.array(
        [
            [0.5, 0.25],
            [0.5, 0.75],
        ]
    )

    np.testing.assert_allclose(kernel.matrix, expected)


def test_normalization_clips_tiny_negative_roundoff() -> None:
    weights = np.array(
        [
            [1.0, -1e-12],
            [1.0, 1.0],
        ]
    )

    kernel = StochasticKernel(
        weights,
        normalize=True,
        tol=1e-10,
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    np.testing.assert_allclose(kernel.matrix, expected)


def test_normalization_rejects_material_negative_entries() -> None:
    weights = np.array(
        [
            [1.1, -0.1],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(ValueError, match="negative"):
        StochasticKernel(
            weights,
            normalize=True,
        )


def test_normalization_rejects_zero_mass_row() -> None:
    weights = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="zero-mass row"):
        StochasticKernel(
            weights,
            normalize=True,
        )


def test_normalization_rejects_zero_mass_column() -> None:
    weights = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="zero-mass column"):
        StochasticKernel(
            weights,
            convention="column",
            normalize=True,
        )


# ============================================================================
# State lookup
# ============================================================================


def test_index_of_returns_state_index(
    labeled_kernel: StochasticKernel,
) -> None:
    assert labeled_kernel.index_of("a") == 0
    assert labeled_kernel.index_of("b") == 1
    assert labeled_kernel.index_of("c") == 2


def test_index_of_rejects_unknown_state(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(KeyError, match="Unknown state"):
        labeled_kernel.index_of("missing")


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (np.int64(1), "b"),
    ],
)
def test_state_at_returns_label(
    labeled_kernel: StochasticKernel,
    index,
    expected,
) -> None:
    assert labeled_kernel.state_at(index) == expected


@pytest.mark.parametrize(
    "index",
    [
        True,
        False,
        1.5,
        "1",
        None,
    ],
)
def test_state_at_rejects_noninteger_index(
    labeled_kernel: StochasticKernel,
    index,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        labeled_kernel.state_at(index)


@pytest.mark.parametrize("index", [-1, 3, 100])
def test_state_at_rejects_out_of_bounds_index(
    labeled_kernel: StochasticKernel,
    index: int,
) -> None:
    with pytest.raises(IndexError, match="outside"):
        labeled_kernel.state_at(index)


# ============================================================================
# Orientation
# ============================================================================


def test_row_matrix_for_row_kernel_returns_internal_view(
    labeled_kernel: StochasticKernel,
) -> None:
    row = labeled_kernel.row_matrix()

    assert row is labeled_kernel.matrix
    assert_read_only(row)


def test_column_matrix_for_row_kernel_returns_transpose(
    labeled_kernel: StochasticKernel,
) -> None:
    column = labeled_kernel.column_matrix()

    np.testing.assert_allclose(
        column,
        labeled_kernel.matrix.T,
    )
    assert_read_only(column)


def test_row_matrix_for_column_kernel(
    row_matrix: np.ndarray,
) -> None:
    kernel = StochasticKernel(
        row_matrix.T,
        convention="column",
    )

    row = kernel.row_matrix()

    np.testing.assert_allclose(row, row_matrix)
    assert_read_only(row)


def test_column_matrix_for_column_kernel(
    column_matrix: np.ndarray,
) -> None:
    kernel = StochasticKernel(
        column_matrix,
        convention="column",
    )

    column = kernel.column_matrix()

    assert column is kernel.matrix
    assert_read_only(column)


# ============================================================================
# Distributions, probabilities, and supports
# ============================================================================


def test_distribution_returns_transition_vector(
    labeled_kernel: StochasticKernel,
) -> None:
    probabilities = labeled_kernel.distribution("b")

    np.testing.assert_allclose(
        probabilities,
        [0.0, 0.25, 0.75],
    )
    assert_read_only(probabilities)


def test_distribution_can_return_labeled_mapping(
    labeled_kernel: StochasticKernel,
) -> None:
    probabilities = labeled_kernel.distribution(
        "b",
        labeled=True,
    )

    assert probabilities == {
        "a": 0.0,
        "b": 0.25,
        "c": 0.75,
    }

    with pytest.raises(TypeError):
        probabilities["a"] = 1.0


def test_probability_returns_scalar_transition_probability(
    labeled_kernel: StochasticKernel,
) -> None:
    assert labeled_kernel.probability("a", "b") == pytest.approx(0.5)
    assert labeled_kernel.probability("b", "c") == pytest.approx(0.75)
    assert labeled_kernel.probability("c", "c") == pytest.approx(1.0)


def test_probability_rejects_unknown_source(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(KeyError):
        labeled_kernel.probability("missing", "a")


def test_probability_rejects_unknown_target(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(KeyError):
        labeled_kernel.probability("a", "missing")


def test_support_uses_kernel_tolerance(
    labeled_kernel: StochasticKernel,
) -> None:
    assert labeled_kernel.support("a") == ("a", "b")
    assert labeled_kernel.support("b") == ("b", "c")
    assert labeled_kernel.support("c") == ("c",)


def test_support_accepts_custom_threshold() -> None:
    kernel = StochasticKernel(
        [
            [0.01, 0.09, 0.90],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        states=("a", "b", "c"),
    )

    assert kernel.support("a", tol=0.05) == ("b", "c")
    assert kernel.support("a", tol=0.10) == ("c",)


# ============================================================================
# Stochastic checks
# ============================================================================


def test_stochastic_sums_for_row_kernel(
    labeled_kernel: StochasticKernel,
) -> None:
    sums = labeled_kernel.stochastic_sums()

    np.testing.assert_allclose(sums, np.ones(3))
    assert_read_only(sums)


def test_stochastic_sums_for_column_kernel(
    column_matrix: np.ndarray,
) -> None:
    kernel = StochasticKernel(
        column_matrix,
        convention="column",
    )

    sums = kernel.stochastic_sums()

    np.testing.assert_allclose(sums, np.ones(3))
    assert_read_only(sums)


def test_row_sums_are_one_under_both_conventions(
    row_matrix: np.ndarray,
) -> None:
    row_kernel = StochasticKernel(row_matrix)
    column_kernel = StochasticKernel(
        row_matrix.T,
        convention="column",
    )

    np.testing.assert_allclose(
        row_kernel.row_sums(),
        np.ones(3),
    )
    np.testing.assert_allclose(
        column_kernel.row_sums(),
        np.ones(3),
    )


def test_column_sums_are_one_under_both_conventions(
    row_matrix: np.ndarray,
) -> None:
    row_kernel = StochasticKernel(row_matrix)
    column_kernel = StochasticKernel(
        row_matrix.T,
        convention="column",
    )

    np.testing.assert_allclose(
        row_kernel.column_sums(),
        np.ones(3),
    )
    np.testing.assert_allclose(
        column_kernel.column_sums(),
        np.ones(3),
    )


def test_is_normalized_returns_true(
    labeled_kernel: StochasticKernel,
) -> None:
    assert labeled_kernel.is_normalized() is True


def test_identity_kernel_is_deterministic() -> None:
    kernel = StochasticKernel.identity(4)

    assert kernel.is_deterministic() is True


def test_permutation_kernel_is_deterministic() -> None:
    kernel = StochasticKernel(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert kernel.is_deterministic() is True


def test_nondeterministic_kernel_is_not_deterministic(
    labeled_kernel: StochasticKernel,
) -> None:
    assert labeled_kernel.is_deterministic() is False


# ============================================================================
# Identity factory
# ============================================================================


def test_identity_factory_from_size() -> None:
    kernel = StochasticKernel.identity(3)

    np.testing.assert_allclose(kernel.matrix, np.eye(3))
    assert kernel.states == (0, 1, 2)
    assert kernel.name == "IdentityKernel"
    assert kernel.metadata["kernel_type"] == "identity"


def test_identity_factory_from_states() -> None:
    kernel = StochasticKernel.identity(
        states=("x", "y"),
    )

    np.testing.assert_allclose(kernel.matrix, np.eye(2))
    assert kernel.states == ("x", "y")


def test_identity_factory_supports_column_convention() -> None:
    kernel = StochasticKernel.identity(
        3,
        convention="column",
    )

    assert kernel.convention == "column"
    np.testing.assert_allclose(kernel.matrix, np.eye(3))


def test_identity_factory_rejects_missing_size_and_states() -> None:
    with pytest.raises(ValueError, match="Either size or states"):
        StochasticKernel.identity()


@pytest.mark.parametrize("size", [True, False, 1.5, "3"])
def test_identity_factory_rejects_noninteger_size(size) -> None:
    with pytest.raises(TypeError, match="size"):
        StochasticKernel.identity(size)


@pytest.mark.parametrize("size", [0, -1, -10])
def test_identity_factory_rejects_nonpositive_size(size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        StochasticKernel.identity(size)


def test_identity_factory_rejects_mismatched_size_and_states() -> None:
    with pytest.raises(ValueError, match="match"):
        StochasticKernel.identity(
            3,
            states=("a", "b"),
        )


# ============================================================================
# Uniform factory
# ============================================================================


def test_uniform_factory_from_size() -> None:
    kernel = StochasticKernel.uniform(3)

    expected = np.full((3, 3), 1.0 / 3.0)

    np.testing.assert_allclose(kernel.matrix, expected)
    assert kernel.name == "UniformKernel"
    assert kernel.metadata["kernel_type"] == "uniform"
    assert kernel.is_deterministic() is False


def test_uniform_factory_from_states() -> None:
    kernel = StochasticKernel.uniform(
        states=("x", "y"),
    )

    np.testing.assert_allclose(
        kernel.matrix,
        np.full((2, 2), 0.5),
    )
    assert kernel.states == ("x", "y")


def test_uniform_factory_supports_column_convention() -> None:
    kernel = StochasticKernel.uniform(
        3,
        convention="column",
    )

    assert kernel.convention == "column"
    np.testing.assert_allclose(
        np.sum(kernel.matrix, axis=0),
        np.ones(3),
    )


# ============================================================================
# Absorbing factory
# ============================================================================


def test_absorbing_factory_with_one_target() -> None:
    kernel = StochasticKernel.absorbing(
        "c",
        states=("a", "b", "c"),
    )

    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(kernel.row_matrix(), expected)
    assert kernel.metadata["kernel_type"] == "absorbing"
    assert kernel.metadata["absorbing_states"] == ("c",)


def test_absorbing_factory_with_multiple_targets() -> None:
    kernel = StochasticKernel.absorbing(
        ("b", "c"),
        states=("a", "b", "c"),
    )

    expected = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(kernel.row_matrix(), expected)


def test_absorbing_factory_supports_column_convention() -> None:
    kernel = StochasticKernel.absorbing(
        "c",
        states=("a", "b", "c"),
        convention="column",
    )

    expected_row = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert kernel.convention == "column"
    np.testing.assert_allclose(
        kernel.row_matrix(),
        expected_row,
    )


def test_absorbing_factory_rejects_empty_states() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        StochasticKernel.absorbing(
            "a",
            states=(),
        )


def test_absorbing_factory_rejects_empty_targets() -> None:
    with pytest.raises(ValueError, match="At least one"):
        StochasticKernel.absorbing(
            (),
            states=("a", "b"),
        )


def test_absorbing_factory_rejects_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        StochasticKernel.absorbing(
            ("a", "a"),
            states=("a", "b"),
        )


def test_absorbing_factory_rejects_unknown_target() -> None:
    with pytest.raises(KeyError, match="Unknown absorbing"):
        StochasticKernel.absorbing(
            "missing",
            states=("a", "b"),
        )


# ============================================================================
# Convention conversion
# ============================================================================


def test_with_convention_transposes_representation(
    labeled_kernel: StochasticKernel,
) -> None:
    converted = labeled_kernel.with_convention("column")

    assert converted.convention == "column"
    np.testing.assert_allclose(
        converted.matrix,
        labeled_kernel.matrix.T,
    )
    np.testing.assert_allclose(
        converted.row_matrix(),
        labeled_kernel.row_matrix(),
    )


def test_with_convention_round_trip_preserves_kernel(
    labeled_kernel: StochasticKernel,
) -> None:
    round_trip = (
        labeled_kernel
        .with_convention("column")
        .with_convention("row")
    )

    np.testing.assert_allclose(
        round_trip.matrix,
        labeled_kernel.matrix,
    )
    assert round_trip.states == labeled_kernel.states


def test_with_same_convention_returns_independent_copy(
    labeled_kernel: StochasticKernel,
) -> None:
    copied = labeled_kernel.with_convention("row")

    assert copied is not labeled_kernel
    np.testing.assert_allclose(
        copied.matrix,
        labeled_kernel.matrix,
    )


# ============================================================================
# Relabeling and copying
# ============================================================================


def test_relabel_changes_only_state_labels(
    labeled_kernel: StochasticKernel,
) -> None:
    relabeled = labeled_kernel.relabel(
        ("x", "y", "z"),
    )

    assert relabeled.states == ("x", "y", "z")
    np.testing.assert_allclose(
        relabeled.matrix,
        labeled_kernel.matrix,
    )
    assert relabeled.metadata["previous_states"] == (
        "a",
        "b",
        "c",
    )


def test_relabel_accepts_custom_name(
    labeled_kernel: StochasticKernel,
) -> None:
    relabeled = labeled_kernel.relabel(
        ("x", "y", "z"),
        name="relabeled",
    )

    assert relabeled.name == "relabeled"


def test_relabel_rejects_invalid_labels(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(ValueError):
        labeled_kernel.relabel(("x", "y"))


def test_copy_returns_independent_kernel(
    labeled_kernel: StochasticKernel,
) -> None:
    copied = labeled_kernel.copy()

    assert copied is not labeled_kernel
    assert copied.matrix is not labeled_kernel.matrix

    np.testing.assert_allclose(
        copied.matrix,
        labeled_kernel.matrix,
    )


def test_copy_can_update_name_and_metadata(
    labeled_kernel: StochasticKernel,
) -> None:
    copied = labeled_kernel.copy(
        name="copied",
        metadata={"version": 2},
    )

    assert copied.name == "copied"
    assert copied.metadata["purpose"] == "testing"
    assert copied.metadata["version"] == 2


# ============================================================================
# Explicit normalization method
# ============================================================================


def test_normalize_returns_equivalent_kernel(
    labeled_kernel: StochasticKernel,
) -> None:
    normalized = labeled_kernel.normalize()

    np.testing.assert_allclose(
        normalized.matrix,
        labeled_kernel.matrix,
    )
    assert normalized.name == "normalize(K)"
    assert normalized.metadata["operation"] == "normalize"


# ============================================================================
# Composition
# ============================================================================


def test_compose_applies_other_then_self() -> None:
    first = StochasticKernel(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        states=("a", "b"),
        name="first",
    )

    second = StochasticKernel(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        states=("a", "b"),
        name="second",
    )

    composite = second.compose(first)

    expected = first.row_matrix() @ second.row_matrix()

    np.testing.assert_allclose(
        composite.row_matrix(),
        expected,
    )


def test_then_applies_self_then_other() -> None:
    first = StochasticKernel(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        states=("a", "b"),
    )

    second = StochasticKernel(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        states=("a", "b"),
    )

    result = first.then(second)

    expected = first.row_matrix() @ second.row_matrix()

    np.testing.assert_allclose(
        result.row_matrix(),
        expected,
    )


def test_matmul_delegates_to_compose() -> None:
    first = StochasticKernel(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    second = StochasticKernel(
        [
            [1.0, 0.0],
            [0.25, 0.75],
        ]
    )

    result = second @ first

    np.testing.assert_allclose(
        result.row_matrix(),
        first.row_matrix() @ second.row_matrix(),
    )


def test_composition_preserves_outer_convention() -> None:
    first = StochasticKernel(
        [
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        convention="row",
    )

    second = StochasticKernel(
        np.array(
            [
                [1.0, 0.25],
                [0.0, 0.75],
            ]
        ),
        convention="column",
    )

    result = second.compose(first)

    assert result.convention == "column"

    np.testing.assert_allclose(
        result.row_matrix(),
        first.row_matrix() @ second.row_matrix(),
    )


def test_composition_uses_larger_tolerance() -> None:
    first = StochasticKernel(np.eye(2), tol=1e-12)
    second = StochasticKernel(np.eye(2), tol=1e-8)

    result = first.compose(second)

    assert result.tol == pytest.approx(1e-8)


def test_composition_rejects_non_kernel(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(TypeError, match="StochasticKernel"):
        labeled_kernel.compose(np.eye(3))


def test_then_rejects_non_kernel(
    labeled_kernel: StochasticKernel,
) -> None:
    with pytest.raises(TypeError, match="StochasticKernel"):
        labeled_kernel.then(np.eye(3))


def test_composition_rejects_different_state_spaces() -> None:
    first = StochasticKernel(
        np.eye(2),
        states=("a", "b"),
    )
    second = StochasticKernel(
        np.eye(2),
        states=("x", "y"),
    )

    with pytest.raises(ValueError, match="state spaces"):
        first.compose(second)


def test_composition_rejects_different_state_orderings() -> None:
    first = StochasticKernel(
        np.eye(2),
        states=("a", "b"),
    )
    second = StochasticKernel(
        np.eye(2),
        states=("b", "a"),
    )

    with pytest.raises(ValueError, match="state spaces"):
        first.compose(second)


# ============================================================================
# Powers
# ============================================================================


def test_zero_power_is_identity(
    labeled_kernel: StochasticKernel,
) -> None:
    result = labeled_kernel.power(0)

    np.testing.assert_allclose(
        result.row_matrix(),
        np.eye(3),
    )
    assert result.states == labeled_kernel.states


def test_first_power_equals_original(
    labeled_kernel: StochasticKernel,
) -> None:
    result = labeled_kernel.power(1)

    np.testing.assert_allclose(
        result.row_matrix(),
        labeled_kernel.row_matrix(),
    )


def test_second_power_matches_matrix_power(
    labeled_kernel: StochasticKernel,
) -> None:
    result = labeled_kernel.power(2)

    expected = np.linalg.matrix_power(
        labeled_kernel.row_matrix(),
        2,
    )

    np.testing.assert_allclose(
        result.row_matrix(),
        expected,
    )


def test_power_preserves_column_convention(
    row_matrix: np.ndarray,
) -> None:
    kernel = StochasticKernel(
        row_matrix.T,
        convention="column",
    )

    result = kernel.power(3)

    assert result.convention == "column"

    np.testing.assert_allclose(
        result.row_matrix(),
        np.linalg.matrix_power(row_matrix, 3),
    )


def test_pow_operator_delegates_to_power(
    labeled_kernel: StochasticKernel,
) -> None:
    result = labeled_kernel**2

    np.testing.assert_allclose(
        result.row_matrix(),
        np.linalg.matrix_power(
            labeled_kernel.row_matrix(),
            2,
        ),
    )


@pytest.mark.parametrize(
    "exponent",
    [
        True,
        False,
        1.5,
        "2",
        None,
    ],
)
def test_power_rejects_noninteger_exponent(
    labeled_kernel: StochasticKernel,
    exponent,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        labeled_kernel.power(exponent)


@pytest.mark.parametrize("exponent", [-1, -2, -100])
def test_power_rejects_negative_exponent(
    labeled_kernel: StochasticKernel,
    exponent: int,
) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        labeled_kernel.power(exponent)


# ============================================================================
# Matrix conversion
# ============================================================================


def test_to_matrix_returns_writable_copy_by_default(
    labeled_kernel: StochasticKernel,
) -> None:
    matrix = labeled_kernel.to_matrix()

    assert matrix is not labeled_kernel.matrix
    assert matrix.flags.writeable is True

    matrix[0, 0] = 123.0

    assert labeled_kernel.matrix[0, 0] == pytest.approx(0.5)


def test_to_matrix_can_return_internal_read_only_matrix(
    labeled_kernel: StochasticKernel,
) -> None:
    matrix = labeled_kernel.to_matrix(copy=False)

    assert matrix is labeled_kernel.matrix
    assert_read_only(matrix)


# ============================================================================
# MarkovOperator conversion
# ============================================================================


def test_from_operator_rejects_non_markov_operator() -> None:
    with pytest.raises(TypeError, match="MarkovOperator"):
        StochasticKernel.from_operator(np.eye(2))


def test_from_operator_preserves_transition_matrix() -> None:
    operator = MarkovOperator(
        matrix=np.array(
            [
                [0.75, 0.25],
                [0.10, 0.90],
            ]
        ),
        convention="row",
        states=("a", "b"),
        name="P",
    )

    kernel = StochasticKernel.from_operator(operator)

    np.testing.assert_allclose(
        kernel.matrix,
        operator.matrix,
    )
    assert kernel.states == ("a", "b")
    assert kernel.convention == "row"
    assert kernel.metadata["source"] == "MarkovOperator"


def test_from_operator_accepts_name_and_metadata_override() -> None:
    operator = MarkovOperator(
        matrix=np.eye(2),
        convention="row",
        states=("a", "b"),
        name="P",
    )

    kernel = StochasticKernel.from_operator(
        operator,
        name="K",
        metadata={"custom": True},
    )

    assert kernel.name == "K"
    assert kernel.metadata["custom"] is True


def test_from_operator_accepts_tolerance_override() -> None:
    operator = MarkovOperator(
        matrix=np.eye(2),
        convention="row",
        states=("a", "b"),
        name="P",
    )

    kernel = StochasticKernel.from_operator(
        operator,
        tol=1e-7,
    )

    assert kernel.tol == pytest.approx(1e-7)


def test_to_operator_preserves_transition_data(
    labeled_kernel: StochasticKernel,
) -> None:
    operator = labeled_kernel.to_operator()

    assert isinstance(operator, MarkovOperator)

    np.testing.assert_allclose(
        operator.matrix,
        labeled_kernel.matrix,
    )

    assert operator.convention == labeled_kernel.convention
    assert operator.states == labeled_kernel.states
    assert operator.metadata["source"] == "StochasticKernel"


def test_kernel_operator_round_trip(
    labeled_kernel: StochasticKernel,
) -> None:
    operator = labeled_kernel.to_operator()
    recovered = StochasticKernel.from_operator(operator)

    np.testing.assert_allclose(
        recovered.row_matrix(),
        labeled_kernel.row_matrix(),
    )

    assert recovered.states == labeled_kernel.states
    assert recovered.convention == labeled_kernel.convention


# ============================================================================
# Summary and representation
# ============================================================================


def test_summary_contains_expected_fields(
    labeled_kernel: StochasticKernel,
) -> None:
    summary = labeled_kernel.summary()

    assert summary["name"] == "K"
    assert summary["shape"] == (3, 3)
    assert summary["size"] == 3
    assert summary["states"] == ("a", "b", "c")
    assert summary["convention"] == "row"
    assert summary["is_normalized"] is True
    assert summary["is_deterministic"] is False
    assert summary["minimum_probability"] == pytest.approx(0.0)
    assert summary["maximum_probability"] == pytest.approx(1.0)
    assert summary["metadata"] == {"purpose": "testing"}


def test_summary_returns_metadata_copy(
    labeled_kernel: StochasticKernel,
) -> None:
    summary = labeled_kernel.summary()

    summary["metadata"]["changed"] = True

    assert "changed" not in labeled_kernel.metadata


def test_repr_contains_class_name_and_basic_fields(
    labeled_kernel: StochasticKernel,
) -> None:
    representation = repr(labeled_kernel)

    assert "StochasticKernel" in representation
    assert "name='K'" in representation
    assert "size=3" in representation
    assert "convention='row'" in representation


# ============================================================================
# Single-state edge cases
# ============================================================================


def test_single_state_kernel() -> None:
    kernel = StochasticKernel(
        [[1.0]],
        states=("only",),
    )

    assert kernel.size == 1
    assert kernel.probability("only", "only") == pytest.approx(1.0)
    assert kernel.support("only") == ("only",)
    assert kernel.is_deterministic() is True


def test_single_state_uniform_kernel() -> None:
    kernel = StochasticKernel.uniform(
        states=("only",),
    )

    np.testing.assert_allclose(kernel.matrix, [[1.0]])
    assert kernel.is_deterministic() is True


def test_single_state_absorbing_kernel() -> None:
    kernel = StochasticKernel.absorbing(
        "only",
        states=("only",),
    )

    np.testing.assert_allclose(kernel.matrix, [[1.0]])
