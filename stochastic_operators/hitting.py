"""
stochastic_operators.hitting
============================

First-hitting analysis for finite-state discrete-time Markov operators and
continuous-time Markov generators.

This module provides:

- target-set normalization,
- graph reachability to a target set,
- eventual hitting probabilities,
- state-specific hitting probabilities,
- almost-sure hitting diagnostics,
- mean first-hitting times,
- state-specific mean hitting times,
- concise hitting-analysis summaries.

All internal calculations use a row-oriented representation, regardless of
the convention used by the underlying stochastic model.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np

from operator_core import (
    OperatorError,
    readonly_array,
)

from .generators import MarkovGenerator
from .markov import MarkovOperator


# ===========================================================================
# Supported Models
# ===========================================================================

_SUPPORTED_MODEL_TYPES = (
    MarkovOperator,
    MarkovGenerator,
)


# ===========================================================================
# Validation Helpers
# ===========================================================================

def _validate_model(
    model,
) -> MarkovOperator | MarkovGenerator:
    """
    Validate and return a supported stochastic model.
    """

    if not isinstance(
        model,
        _SUPPORTED_MODEL_TYPES,
    ):
        raise OperatorError(
            "model must be a MarkovOperator or MarkovGenerator."
        )

    return model


def _validate_nonnegative_scalar(
    value,
    *,
    name: str,
) -> float:
    """
    Validate a finite nonnegative scalar.
    """

    if (
        isinstance(value, bool)
        or not np.isscalar(value)
    ):
        raise OperatorError(
            f"{name} must be a nonnegative scalar."
        )

    result = float(value)

    if (
        not np.isfinite(result)
        or result < 0.0
    ):
        raise OperatorError(
            f"{name} must be finite and nonnegative."
        )

    return result


# ===========================================================================
# Hitting Analyzer
# ===========================================================================

class HittingAnalyzer:
    """
    Analyze first-hitting behavior for a finite-state stochastic model.

    Parameters
    ----------
    model
        A MarkovOperator or MarkovGenerator.

    tol
        Optional numerical tolerance. If omitted, the tolerance of the
        underlying stochastic model is used.

    Notes
    -----
    For a discrete-time row-stochastic transition matrix ``P``, the hitting
    probability vector ``h`` for target set ``T`` satisfies

        h_i = 1,                                  i in T,

        h_i = sum_j P_ij h_j,                     i not in T.

    For a continuous-time row generator ``Q``, it satisfies

        h_i = 1,                                  i in T,

        sum_j Q_ij h_j = 0,                       i not in T.

    Mean hitting times are finite only for states that hit the target set
    with probability one. States without almost-sure hitting are assigned
    ``np.inf``.
    """

    def __init__(
        self,
        model,
        *,
        tol: float | None = None,
    ):
        model = _validate_model(
            model
        )

        if tol is None:
            tolerance = float(
                model.tol
            )
        else:
            tolerance = (
                _validate_nonnegative_scalar(
                    tol,
                    name="tol",
                )
            )

        object.__setattr__(
            self,
            "model",
            model,
        )

        object.__setattr__(
            self,
            "tol",
            tolerance,
        )

    # -----------------------------------------------------------------------
    # Basic Properties
    # -----------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """
        Return the number of states.
        """

        return self.model.dimension

    @property
    def states(self) -> tuple:
        """
        Return the state labels.
        """

        return self.model.states

    @property
    def is_discrete_time(self) -> bool:
        """
        Return whether the model is discrete-time.
        """

        return isinstance(
            self.model,
            MarkovOperator,
        )

    @property
    def is_continuous_time(self) -> bool:
        """
        Return whether the model is continuous-time.
        """

        return isinstance(
            self.model,
            MarkovGenerator,
        )

    # -----------------------------------------------------------------------
    # Convention-Independent Matrices
    # -----------------------------------------------------------------------

    def row_oriented_matrix(
        self,
    ) -> np.ndarray:
        """
        Return the transition matrix or generator in row convention.
        """

        matrix = self.model.matrix

        if self.is_discrete_time:
            row_oriented = (
                matrix
                if self.model.is_row_stochastic
                else matrix.T
            )
        else:
            row_oriented = (
                matrix
                if self.model.is_row_generator
                else matrix.T
            )

        return readonly_array(
            row_oriented,
            name="row-oriented hitting matrix",
            ndim=2,
        )

    def adjacency_matrix(
        self,
    ) -> np.ndarray:
        """
        Return the directed adjacency matrix of possible transitions.

        For continuous-time generators, only positive off-diagonal rates
        define edges. Generator diagonals are excluded.
        """

        matrix = self.row_oriented_matrix()

        adjacency = np.asarray(
            matrix > self.tol,
            dtype=bool,
        )

        if self.is_continuous_time:
            np.fill_diagonal(
                adjacency,
                False,
            )

        return readonly_array(
            adjacency,
            name="hitting adjacency matrix",
            ndim=2,
        )

    # -----------------------------------------------------------------------
    # State and Target Normalization
    # -----------------------------------------------------------------------

    def state_index(
        self,
        state,
    ) -> int:
        """
        Return the integer index of a state label.
        """

        try:
            return int(
                self.model.state_index(
                    state
                )
            )
        except Exception as error:
            raise OperatorError(
                f"unknown state: {state!r}."
            ) from error

    def _is_single_state(
        self,
        value,
    ) -> bool:
        """
        Return whether a value is directly recognized as one state label.
        """

        try:
            self.model.state_index(
                value
            )
        except Exception:
            return False

        return True

    def target_indices(
        self,
        targets,
    ) -> tuple[int, ...]:
        """
        Normalize one or more target states into sorted unique indices.

        A scalar state label is accepted directly. Otherwise, ``targets``
        must be a nonempty iterable of valid state labels.
        """

        if self._is_single_state(
            targets
        ):
            return (
                self.state_index(
                    targets
                ),
            )

        if isinstance(
            targets,
            (str, bytes),
        ):
            raise OperatorError(
                f"unknown target state: {targets!r}."
            )

        if not isinstance(
            targets,
            Iterable,
        ):
            raise OperatorError(
                "targets must be a state or a nonempty iterable of states."
            )

        target_list = list(
            targets
        )

        if not target_list:
            raise OperatorError(
                "targets must contain at least one state."
            )

        indices = []

        for target in target_list:
            indices.append(
                self.state_index(
                    target
                )
            )

        return tuple(
            sorted(
                set(indices)
            )
        )

    def target_states(
        self,
        targets,
    ) -> tuple:
        """
        Return normalized target-state labels.
        """

        return tuple(
            self.states[index]
            for index in self.target_indices(
                targets
            )
        )

    def target_mask(
        self,
        targets,
    ) -> np.ndarray:
        """
        Return a Boolean mask for the target set.
        """

        mask = np.zeros(
            self.dimension,
            dtype=bool,
        )

        mask[
            list(
                self.target_indices(
                    targets
                )
            )
        ] = True

        return readonly_array(
            mask,
            name="target mask",
            ndim=1,
        )

    # -----------------------------------------------------------------------
    # Reachability
    # -----------------------------------------------------------------------

    def reachable_mask(
        self,
        targets,
    ) -> np.ndarray:
        """
        Return states from which at least one target is graph-reachable.

        Reachability is computed by traversing the reversed transition graph
        outward from the target set.
        """

        target_indices = (
            self.target_indices(
                targets
            )
        )

        adjacency = np.asarray(
            self.adjacency_matrix()
        )

        reverse_adjacency = (
            adjacency.T
        )

        reachable = np.zeros(
            self.dimension,
            dtype=bool,
        )

        queue = deque()

        for index in target_indices:
            reachable[index] = True
            queue.append(index)

        while queue:
            current = queue.popleft()

            predecessors = np.flatnonzero(
                reverse_adjacency[current]
            )

            for predecessor in predecessors:
                predecessor = int(
                    predecessor
                )

                if not reachable[
                    predecessor
                ]:
                    reachable[
                        predecessor
                    ] = True

                    queue.append(
                        predecessor
                    )

        return readonly_array(
            reachable,
            name="target-reachable mask",
            ndim=1,
        )

    def reachable_states(
        self,
        targets,
    ) -> tuple:
        """
        Return labels of states that can reach the target set.
        """

        mask = self.reachable_mask(
            targets
        )

        return tuple(
            self.states[index]
            for index in np.flatnonzero(
                mask
            )
        )

    def can_reach(
        self,
        start,
        targets,
    ) -> bool:
        """
        Return whether a start state can reach the target set.
        """

        start_index = self.state_index(
            start
        )

        return bool(
            self.reachable_mask(
                targets
            )[start_index]
        )

    # -----------------------------------------------------------------------
    # Internal Linear Solver
    # -----------------------------------------------------------------------

    def _solve_linear_system(
        self,
        matrix: np.ndarray,
        right_hand_side: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        """
        Solve a linear system and validate its residual.

        A least-squares fallback is used for numerically singular systems,
        but the resulting residual must remain within a tolerance-scaled
        numerical threshold.
        """

        if matrix.size == 0:
            return np.empty(
                0,
                dtype=float,
            )

        try:
            solution = np.linalg.solve(
                matrix,
                right_hand_side,
            )
        except np.linalg.LinAlgError:
            solution, _, _, _ = (
                np.linalg.lstsq(
                    matrix,
                    right_hand_side,
                    rcond=None,
                )
            )

        residual = (
            matrix @ solution
            - right_hand_side
        )

        residual_norm = float(
            np.linalg.norm(
                residual,
                ord=np.inf,
            )
        )

        scale = max(
            1.0,
            float(
                np.linalg.norm(
                    matrix,
                    ord=np.inf,
                )
            ),
            float(
                np.linalg.norm(
                    right_hand_side,
                    ord=np.inf,
                )
            ),
        )

        residual_threshold = max(
            self.tol,
            np.sqrt(
                np.finfo(float).eps
            ),
        ) * scale

        if residual_norm > residual_threshold:
            raise OperatorError(
                f"could not reliably solve the {name} system."
            )

        return np.asarray(
            solution,
            dtype=float,
        )

    # -----------------------------------------------------------------------
    # Eventual Hitting Probabilities
    # -----------------------------------------------------------------------

    def hitting_probabilities(
        self,
        targets,
    ) -> np.ndarray:
        """
        Return eventual probabilities of hitting the target set.

        Target states receive probability one. States that cannot reach the
        target set receive probability zero.
        """

        target_mask = np.asarray(
            self.target_mask(
                targets
            )
        )

        reachable_mask = np.asarray(
            self.reachable_mask(
                targets
            )
        )

        probabilities = np.zeros(
            self.dimension,
            dtype=float,
        )

        probabilities[
            target_mask
        ] = 1.0

        unknown_mask = (
            reachable_mask
            & ~target_mask
        )

        unknown_indices = np.flatnonzero(
            unknown_mask
        )

        if unknown_indices.size == 0:
            return readonly_array(
                probabilities,
                name="hitting probabilities",
                ndim=1,
            )

        target_indices = np.flatnonzero(
            target_mask
        )

        matrix = np.asarray(
            self.row_oriented_matrix()
        )

        if self.is_discrete_time:
            transition_unknown = matrix[
                np.ix_(
                    unknown_indices,
                    unknown_indices,
                )
            ]

            transition_target = matrix[
                np.ix_(
                    unknown_indices,
                    target_indices,
                )
            ]

            system_matrix = (
                np.eye(
                    len(unknown_indices),
                    dtype=float,
                )
                - transition_unknown
            )

            right_hand_side = (
                transition_target
                @ np.ones(
                    len(target_indices),
                    dtype=float,
                )
            )

        else:
            generator_unknown = matrix[
                np.ix_(
                    unknown_indices,
                    unknown_indices,
                )
            ]

            generator_target = matrix[
                np.ix_(
                    unknown_indices,
                    target_indices,
                )
            ]

            system_matrix = (
                generator_unknown
            )

            right_hand_side = -(
                generator_target
                @ np.ones(
                    len(target_indices),
                    dtype=float,
                )
            )

        solution = self._solve_linear_system(
            system_matrix,
            right_hand_side,
            name="hitting-probability",
        )

        probabilities[
            unknown_indices
        ] = solution

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        probabilities[
            np.abs(
                probabilities
            ) <= self.tol
        ] = 0.0

        probabilities[
            np.abs(
                probabilities - 1.0
            ) <= self.tol
        ] = 1.0

        return readonly_array(
            probabilities,
            name="hitting probabilities",
            ndim=1,
        )

    def hitting_probability(
        self,
        start,
        targets,
    ) -> float:
        """
        Return the eventual target-hitting probability from one state.
        """

        start_index = self.state_index(
            start
        )

        return float(
            self.hitting_probabilities(
                targets
            )[start_index]
        )

    def almost_sure_mask(
        self,
        targets,
    ) -> np.ndarray:
        """
        Return states that hit the target set with probability one.
        """

        probabilities = (
            self.hitting_probabilities(
                targets
            )
        )

        mask = np.isclose(
            probabilities,
            1.0,
            atol=self.tol,
            rtol=self.tol,
        )

        return readonly_array(
            mask,
            name="almost-sure hitting mask",
            ndim=1,
        )

    def almost_sure_states(
        self,
        targets,
    ) -> tuple:
        """
        Return labels of states that hit the target almost surely.
        """

        mask = self.almost_sure_mask(
            targets
        )

        return tuple(
            self.states[index]
            for index in np.flatnonzero(
                mask
            )
        )

    def is_almost_surely_hitting(
        self,
        start,
        targets,
    ) -> bool:
        """
        Return whether one start state hits the target with probability one.
        """

        return bool(
            np.isclose(
                self.hitting_probability(
                    start,
                    targets,
                ),
                1.0,
                atol=self.tol,
                rtol=self.tol,
            )
        )

    # -----------------------------------------------------------------------
    # Mean First-Hitting Times
    # -----------------------------------------------------------------------

    def mean_hitting_times(
        self,
        targets,
    ) -> np.ndarray:
        """
        Return mean first-hitting times for all states.

        Target states receive zero. States that do not hit the target set
        almost surely receive ``np.inf``.

        For discrete time, time is measured in transition steps.

        For continuous time, time is measured in the units associated with
        the generator rates.
        """

        target_mask = np.asarray(
            self.target_mask(
                targets
            )
        )

        almost_sure_mask = np.asarray(
            self.almost_sure_mask(
                targets
            )
        )

        times = np.full(
            self.dimension,
            np.inf,
            dtype=float,
        )

        times[
            target_mask
        ] = 0.0

        finite_unknown_mask = (
            almost_sure_mask
            & ~target_mask
        )

        finite_indices = np.flatnonzero(
            finite_unknown_mask
        )

        if finite_indices.size == 0:
            return readonly_array(
                times,
                name="mean hitting times",
                ndim=1,
            )

        matrix = np.asarray(
            self.row_oriented_matrix()
        )

        if self.is_discrete_time:
            restricted_transition = matrix[
                np.ix_(
                    finite_indices,
                    finite_indices,
                )
            ]

            system_matrix = (
                np.eye(
                    len(finite_indices),
                    dtype=float,
                )
                - restricted_transition
            )

            right_hand_side = np.ones(
                len(finite_indices),
                dtype=float,
            )

        else:
            restricted_generator = matrix[
                np.ix_(
                    finite_indices,
                    finite_indices,
                )
            ]

            system_matrix = -(
                restricted_generator
            )

            right_hand_side = np.ones(
                len(finite_indices),
                dtype=float,
            )

        solution = self._solve_linear_system(
            system_matrix,
            right_hand_side,
            name="mean-hitting-time",
        )

        solution[
            np.abs(solution) <= self.tol
        ] = 0.0

        if np.any(
            solution < -self.tol
        ):
            raise OperatorError(
                "mean-hitting-time solver produced negative values."
            )

        times[
            finite_indices
        ] = np.maximum(
            solution,
            0.0,
        )

        return readonly_array(
            times,
            name="mean hitting times",
            ndim=1,
        )

    def mean_hitting_time(
        self,
        start,
        targets,
    ) -> float:
        """
        Return the mean first-hitting time from one state.
        """

        start_index = self.state_index(
            start
        )

        return float(
            self.mean_hitting_times(
                targets
            )[start_index]
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(
        self,
        targets,
        *,
        start=None,
    ) -> dict:
        """
        Return target-set hitting diagnostics.

        Parameters
        ----------
        targets
            One target state or a nonempty iterable of target states.

        start
            Optional start state for state-specific diagnostics.
        """

        normalized_targets = (
            self.target_states(
                targets
            )
        )

        probabilities = (
            self.hitting_probabilities(
                normalized_targets
            )
        )

        times = self.mean_hitting_times(
            normalized_targets
        )

        result = {
            "model": self.model.name,
            "model_type": (
                "discrete_time"
                if self.is_discrete_time
                else "continuous_time"
            ),
            "dimension": self.dimension,
            "states": self.states,
            "targets": normalized_targets,
            "reachable_states":
                self.reachable_states(
                    normalized_targets
                ),
            "almost_sure_states":
                self.almost_sure_states(
                    normalized_targets
                ),
            "hitting_probabilities":
                tuple(
                    probabilities.tolist()
                ),
            "mean_hitting_times":
                tuple(
                    times.tolist()
                ),
        }

        if start is not None:
            start_index = self.state_index(
                start
            )

            result.update({
                "start":
                    self.states[
                        start_index
                    ],
                "start_can_reach":
                    bool(
                        self.reachable_mask(
                            normalized_targets
                        )[start_index]
                    ),
                "start_hitting_probability":
                    float(
                        probabilities[
                            start_index
                        ]
                    ),
                "start_almost_sure":
                    bool(
                        np.isclose(
                            probabilities[
                                start_index
                            ],
                            1.0,
                            atol=self.tol,
                            rtol=self.tol,
                        )
                    ),
                "start_mean_hitting_time":
                    float(
                        times[
                            start_index
                        ]
                    ),
            })

        return result
