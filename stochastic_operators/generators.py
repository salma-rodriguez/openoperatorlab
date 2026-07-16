"""
stochastic_operators.generators
===============================

Infinitesimal generators for finite-state continuous-time Markov systems.

This module defines conservative Markov generators, their transition
semigroups, and continuous-time probability evolution.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from operator_core import (
    DimensionMismatchError,
    NonSquareOperatorError,
    OperatorError,
    readonly_array,
)
from .markov import MarkovOperator
from .operators import (
    StochasticConvention,
    _CONVENTION_ALIASES,
)


# ===========================================================================
# Shared Helpers
# ===========================================================================

def _validate_nonnegative_time(
    t,
    *,
    name: str = "t",
) -> float:
    """
    Validate and return a finite nonnegative time parameter.
    """

    if isinstance(t, bool) or not np.isscalar(t):
        raise OperatorError(
            f"{name} must be a nonnegative scalar."
        )

    value = float(t)

    if not np.isfinite(value):
        raise OperatorError(
            f"{name} must be finite."
        )

    if value < 0:
        raise OperatorError(
            f"{name} must be nonnegative."
        )

    return value


def _validate_generator_states(
    states,
    *,
    dimension: int,
) -> tuple:
    """
    Validate state labels for a Markov generator.
    """

    if states is None:
        return tuple(range(dimension))

    labels = tuple(states)

    if len(labels) != dimension:
        raise DimensionMismatchError(
            "number of state labels must match generator dimension."
        )

    if len(set(labels)) != len(labels):
        raise OperatorError(
            "state labels must be unique."
        )

    return labels


# ===========================================================================
# Markov Generator
# ===========================================================================

class MarkovGenerator:
    """
    Finite-state continuous-time Markov generator.

    Parameters
    ----------
    matrix
        Square real generator matrix.

    states
        Optional unique state labels.

    convention
        Row or column convention.

    tol
        Numerical tolerance used in generator validation.

    name
        Human-readable generator name.

    metadata
        Optional user metadata.

    Notes
    -----
    A conservative row generator satisfies

        q_ij >= 0 for i != j,
        sum_j q_ij = 0.

    A conservative column generator satisfies

        q_ij >= 0 for i != j,
        sum_i q_ij = 0.
    """

    def __init__(
        self,
        matrix,
        *,
        states=None,
        convention: StochasticConvention | str = StochasticConvention.ROW,
        tol: float = 1e-10,
        name: str = "MarkovGenerator",
        metadata: dict | None = None,
    ):
        try:
            convention = _CONVENTION_ALIASES[convention]
        except KeyError as exc:
            raise OperatorError(
                "convention must be 'row' or 'column'."
            ) from exc

        if isinstance(tol, bool) or not np.isscalar(tol):
            raise OperatorError(
                "tol must be a nonnegative scalar."
            )

        tolerance = float(tol)

        if not np.isfinite(tolerance) or tolerance < 0:
            raise OperatorError(
                "tol must be finite and nonnegative."
            )

        array = np.asarray(matrix)

        if array.ndim != 2:
            raise OperatorError(
                "generator matrix must be two-dimensional."
            )

        if array.shape[0] != array.shape[1]:
            raise NonSquareOperatorError(
                "Markov generators must be square."
            )

        if np.iscomplexobj(array):
            raise OperatorError(
                "Markov generators must be real-valued."
            )

        array = np.array(
            array,
            dtype=float,
            copy=True,
        )

        if not np.all(np.isfinite(array)):
            raise OperatorError(
                "generator entries must be finite."
            )

        dimension = array.shape[0]

        off_diagonal = array.copy()
        np.fill_diagonal(
            off_diagonal,
            0.0,
        )

        if np.any(off_diagonal < -tolerance):
            raise OperatorError(
                "off-diagonal generator entries must be nonnegative."
            )

        diagonal = np.diag(array)

        if np.any(diagonal > tolerance):
            raise OperatorError(
                "generator diagonal entries must be nonpositive."
            )

        axis = (
            1
            if convention is StochasticConvention.ROW
            else 0
        )

        sums = np.sum(
            array,
            axis=axis,
        )

        if not np.allclose(
            sums,
            0.0,
            atol=tolerance,
            rtol=tolerance,
        ):
            direction = (
                "rows"
                if convention is StochasticConvention.ROW
                else "columns"
            )

            raise OperatorError(
                f"generator {direction} must sum to zero."
            )

        labels = _validate_generator_states(
            states,
            dimension=dimension,
        )

        state_to_index = {
            state: index
            for index, state in enumerate(labels)
        }

        generator_metadata = {
            "operator": "markov_generator",
            "convention": convention.value,
            "tolerance": tolerance,
            "states": labels,
        }

        if metadata is not None:
            if not isinstance(metadata, dict):
                raise OperatorError(
                    "metadata must be a dictionary."
                )

            generator_metadata.update(
                metadata
            )

        object.__setattr__(
            self,
            "matrix",
            readonly_array(
                array,
                name="generator matrix",
                ndim=2,
            ),
        )
        object.__setattr__(
            self,
            "states",
            labels,
        )
        object.__setattr__(
            self,
            "_state_to_index",
            state_to_index,
        )
        object.__setattr__(
            self,
            "convention",
            convention,
        )
        object.__setattr__(
            self,
            "tol",
            tolerance,
        )
        object.__setattr__(
            self,
            "name",
            name,
        )
        object.__setattr__(
            self,
            "metadata",
            generator_metadata,
        )

    # -----------------------------------------------------------------------
    # Basic Properties
    # -----------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """
        Return the number of states.
        """

        return self.matrix.shape[0]

    @property
    def is_column_generator(self) -> bool:
        """
        Return whether column sums vanish.
        """

        return (
            self.convention
            is StochasticConvention.COLUMN
        )

    @property
    def is_row_generator(self) -> bool:
        """
        Return whether row sums vanish.
        """

        return (
            self.convention
            is StochasticConvention.ROW
        )

    def state_index(
        self,
        state,
    ) -> int:
        """
        Return the index associated with a state label.
        """

        try:
            return self._state_to_index[state]
        except KeyError as exc:
            raise OperatorError(
                f"unknown state label: {state!r}."
            ) from exc

    def state_label(
        self,
        index: int,
    ):
        """
        Return the state label associated with an index.
        """

        if (
            isinstance(index, bool)
            or not isinstance(
                index,
                (int, np.integer),
            )
        ):
            raise OperatorError(
                "index must be a nonnegative integer."
            )

        index = int(index)

        if index < 0 or index >= self.dimension:
            raise OperatorError(
                "state index is outside the valid range."
            )

        return self.states[index]

    # -----------------------------------------------------------------------
    # Rates
    # -----------------------------------------------------------------------

    def exit_rate(
        self,
        state,
    ) -> float:
        """
        Return the total exit rate from a state.
        """

        index = self.state_index(state)

        return float(
            -self.matrix[index, index]
        )

    def holding_rates(self) -> np.ndarray:
        """
        Return all state exit rates.
        """

        return readonly_array(
            -np.diag(self.matrix),
            name="holding rates",
            ndim=1,
        )

    def jump_rate(
        self,
        source,
        target,
    ) -> float:
        """
        Return the instantaneous jump rate from source to target.
        """

        source_index = self.state_index(
            source
        )
        target_index = self.state_index(
            target
        )

        if source_index == target_index:
            return 0.0

        if self.is_row_generator:
            rate = self.matrix[
                source_index,
                target_index,
            ]
        else:
            rate = self.matrix[
                target_index,
                source_index,
            ]

        return float(rate)

    # -----------------------------------------------------------------------
    # Transition Semigroup
    # -----------------------------------------------------------------------

    def transition_matrix(
        self,
        t,
    ) -> np.ndarray:
        """
        Return the transition matrix P(t) = exp(tQ).
        """

        time = _validate_nonnegative_time(
            t
        )

        transition = expm(
            time * self.matrix
        )

        # Remove tiny floating-point artifacts without renormalizing
        # substantive errors.
        transition[
            np.abs(transition) <= self.tol
        ] = 0.0

        if np.any(
            transition < -self.tol
        ):
            raise OperatorError(
                "matrix exponential produced invalid negative probabilities."
            )

        transition[
            transition < 0.0
        ] = 0.0

        axis = (
            1
            if self.is_row_generator
            else 0
        )

        sums = np.sum(
            transition,
            axis=axis,
        )

        if not np.allclose(
            sums,
            1.0,
            atol=max(self.tol, 1e-12),
            rtol=max(self.tol, 1e-12),
        ):
            raise OperatorError(
                "matrix exponential did not produce a stochastic matrix."
            )

        return readonly_array(
            transition,
            name="transition matrix",
            ndim=2,
        )

    def transition_operator(
        self,
        t,
        *,
        name: str | None = None,
    ) -> MarkovOperator:
        """
        Return the discrete transition operator at time t.
        """

        time = _validate_nonnegative_time(
            t
        )

        return MarkovOperator(
            self.transition_matrix(time),
            states=self.states,
            convention=self.convention,
            tol=max(self.tol, 1e-12),
            name=(
                name
                or f"{self.name}@t={time:g}"
            ),
            metadata={
                **self.metadata,
                "source_generator": self.name,
                "time": time,
            },
        )

    def transition_probability(
        self,
        source,
        target,
        *,
        t,
    ) -> float:
        """
        Return the transition probability over elapsed time t.
        """

        transition = self.transition_matrix(
            t
        )

        source_index = self.state_index(
            source
        )
        target_index = self.state_index(
            target
        )

        if self.is_row_generator:
            probability = transition[
                source_index,
                target_index,
            ]
        else:
            probability = transition[
                target_index,
                source_index,
            ]

        return float(probability)

    # -----------------------------------------------------------------------
    # Distribution Evolution
    # -----------------------------------------------------------------------

    def evolve_distribution(
        self,
        distribution,
        *,
        t,
    ) -> np.ndarray:
        """
        Evolve a probability distribution through time t.
        """

        vector = np.asarray(
            distribution,
            dtype=float,
        )

        if vector.ndim != 1:
            raise OperatorError(
                "distribution must be one-dimensional."
            )

        if len(vector) != self.dimension:
            raise DimensionMismatchError(
                "distribution dimension does not match generator."
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise OperatorError(
                "distribution values must be finite."
            )

        if np.any(
            vector < -self.tol
        ):
            raise OperatorError(
                "distribution values must be nonnegative."
            )

        if not np.isclose(
            np.sum(vector),
            1.0,
            atol=self.tol,
            rtol=self.tol,
        ):
            raise OperatorError(
                "distribution must sum to one."
            )

        transition = self.transition_matrix(
            t
        )

        if self.is_row_generator:
            evolved = vector @ transition
        else:
            evolved = transition @ vector

        evolved[
            np.abs(evolved) <= self.tol
        ] = 0.0

        return readonly_array(
            evolved,
            name="evolved distribution",
            ndim=1,
        )

    def distribution_history(
        self,
        distribution,
        times,
    ) -> np.ndarray:
        """
        Evaluate a distribution at a one-dimensional sequence of times.

        The output has shape ``(len(times), dimension)``.
        """

        time_values = np.asarray(
            times,
            dtype=float,
        )

        if time_values.ndim != 1:
            raise OperatorError(
                "times must be one-dimensional."
            )

        if time_values.size == 0:
            raise OperatorError(
                "times cannot be empty."
            )

        if not np.all(
            np.isfinite(time_values)
        ):
            raise OperatorError(
                "times must contain finite values."
            )

        if np.any(time_values < 0):
            raise OperatorError(
                "times must be nonnegative."
            )

        history = np.vstack([
            self.evolve_distribution(
                distribution,
                t=time,
            )
            for time in time_values
        ])

        return readonly_array(
            history,
            name="distribution history",
            ndim=2,
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def is_conservative(self) -> bool:
        """
        Check the conservative generator conditions.
        """

        off_diagonal = self.matrix.copy()
        np.fill_diagonal(
            off_diagonal,
            0.0,
        )

        axis = (
            1
            if self.is_row_generator
            else 0
        )

        return bool(
            np.all(
                off_diagonal >= -self.tol
            )
            and np.allclose(
                np.sum(
                    self.matrix,
                    axis=axis,
                ),
                0.0,
                atol=self.tol,
                rtol=self.tol,
            )
        )

    def summary(self) -> dict:
        """
        Return generator metadata and rate diagnostics.
        """

        holding_rates = (
            self.holding_rates()
        )

        return {
            "name": self.name,
            "dimension": self.dimension,
            "states": self.states,
            "convention":
                self.convention.value,
            "conservative":
                self.is_conservative(),
            "holding_rates":
                tuple(
                    holding_rates.tolist()
                ),
            "max_exit_rate": float(
                np.max(holding_rates)
            ),
            "min_exit_rate": float(
                np.min(holding_rates)
            ),
        }
