"""
stochastic_operators.stationary
===============================

Stationary-distribution and reversibility analysis for finite-state
discrete-time Markov operators and continuous-time Markov generators.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import lsq_linear

from operator_core import (
    DimensionMismatchError,
    OperatorError,
    readonly_array,
)
from .generators import MarkovGenerator
from .markov import MarkovOperator


# ===========================================================================
# Shared Helpers
# ===========================================================================

_SUPPORTED_OPERATOR_TYPES = (
    MarkovOperator,
    MarkovGenerator,
)


def _validate_stochastic_model(
    operator,
) -> MarkovOperator | MarkovGenerator:
    """
    Validate and return a supported stochastic model.
    """

    if not isinstance(
        operator,
        _SUPPORTED_OPERATOR_TYPES,
    ):
        raise OperatorError(
            "operator must be a MarkovOperator or MarkovGenerator."
        )

    return operator


def _validate_distribution(
    distribution,
    *,
    dimension: int,
    tol: float,
    name: str = "distribution",
) -> np.ndarray:
    """
    Validate and return a probability distribution.
    """

    values = np.asarray(
        distribution,
        dtype=float,
    )

    if values.ndim != 1:
        raise OperatorError(
            f"{name} must be one-dimensional."
        )

    if len(values) != dimension:
        raise DimensionMismatchError(
            f"{name} dimension does not match the stochastic model."
        )

    if not np.all(np.isfinite(values)):
        raise OperatorError(
            f"{name} values must be finite."
        )

    if np.any(values < -tol):
        raise OperatorError(
            f"{name} values must be nonnegative."
        )

    if not np.isclose(
        np.sum(values),
        1.0,
        atol=tol,
        rtol=tol,
    ):
        raise OperatorError(
            f"{name} must sum to one."
        )

    values = np.array(
        values,
        dtype=float,
        copy=True,
    )

    values[
        np.abs(values) <= tol
    ] = 0.0

    return values


# ===========================================================================
# Stationary Analyzer
# ===========================================================================

class StationaryAnalyzer:
    """
    Analyze stationary distributions and reversibility.

    Parameters
    ----------
    operator
        MarkovOperator or MarkovGenerator.

    tol
        Optional numerical tolerance. If omitted, the tolerance of the
        underlying stochastic model is used.
    """

    def __init__(
        self,
        operator,
        *,
        tol: float | None = None,
    ):
        operator = _validate_stochastic_model(
            operator
        )

        if tol is None:
            tolerance = float(
                operator.tol
            )
        else:
            if (
                isinstance(tol, bool)
                or not np.isscalar(tol)
            ):
                raise OperatorError(
                    "tol must be a nonnegative scalar."
                )

            tolerance = float(tol)

            if (
                not np.isfinite(tolerance)
                or tolerance < 0
            ):
                raise OperatorError(
                    "tol must be finite and nonnegative."
                )

        object.__setattr__(
            self,
            "operator",
            operator,
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

        return self.operator.dimension

    @property
    def states(self) -> tuple:
        """
        Return state labels.
        """

        return self.operator.states

    @property
    def is_discrete_time(self) -> bool:
        """
        Return whether the model is a MarkovOperator.
        """

        return isinstance(
            self.operator,
            MarkovOperator,
        )

    @property
    def is_continuous_time(self) -> bool:
        """
        Return whether the model is a MarkovGenerator.
        """

        return isinstance(
            self.operator,
            MarkovGenerator,
        )

    # -----------------------------------------------------------------------
    # Stationary Linear System
    # -----------------------------------------------------------------------

    def stationary_system_matrix(
        self,
    ) -> np.ndarray:
        """
        Return the matrix A for the stationary equation A pi = 0.

        The returned stationary distribution is always represented as
        a column vector internally, independent of row/column convention.
        """

        matrix = self.operator.matrix

        if self.is_discrete_time:
            if self.operator.is_row_stochastic:
                system = (
                    matrix.T
                    - np.eye(
                        self.dimension,
                        dtype=matrix.dtype,
                    )
                )
            else:
                system = (
                    matrix
                    - np.eye(
                        self.dimension,
                        dtype=matrix.dtype,
                    )
                )

        else:
            if self.operator.is_row_generator:
                system = matrix.T
            else:
                system = matrix

        return readonly_array(
            system,
            name="stationary system matrix",
            ndim=2,
        )

    def stationary_space_basis(
        self,
    ) -> np.ndarray:
        """
        Return an orthonormal basis for the stationary nullspace.

        Basis vectors are stored as columns.
        """

        basis = null_space(
            self.stationary_system_matrix(),
            rcond=self.tol,
        )

        return readonly_array(
            basis,
            name="stationary space basis",
            ndim=2,
        )

    def stationary_dimension(
        self,
    ) -> int:
        """
        Return the dimension of the stationary solution space.
        """

        return int(
            self.stationary_space_basis().shape[1]
        )

    def is_unique(
        self,
    ) -> bool:
        """
        Return whether the normalized stationary distribution is unique.
        """

        return (
            self.stationary_dimension()
            == 1
        )

    # -----------------------------------------------------------------------
    # Stationary Distribution
    # -----------------------------------------------------------------------

    def stationary_distribution(
        self,
    ) -> np.ndarray:
        """
        Return one normalized nonnegative stationary distribution.

        For systems with multiple stationary distributions, this method
        returns one feasible solution and ``is_unique()`` remains false.
        """

        system = self.stationary_system_matrix()

        augmented_matrix = np.vstack((
            system,
            np.ones(
                (1, self.dimension),
                dtype=float,
            ),
        ))

        target = np.concatenate((
            np.zeros(
                self.dimension,
                dtype=float,
            ),
            np.ones(
                1,
                dtype=float,
            ),
        ))

        solution = lsq_linear(
            augmented_matrix,
            target,
            bounds=(
                np.zeros(
                    self.dimension,
                    dtype=float,
                ),
                np.full(
                    self.dimension,
                    np.inf,
                    dtype=float,
                ),
            ),
            tol=max(
                self.tol,
                1e-15,
            ),
            lsmr_tol="auto",
        )

        if not solution.success:
            raise OperatorError(
                "could not compute a stationary distribution."
            )

        distribution = solution.x

        distribution[
            np.abs(distribution)
            <= self.tol
        ] = 0.0

        total = float(
            np.sum(distribution)
        )

        if total <= self.tol:
            raise OperatorError(
                "stationary solver produced a zero distribution."
            )

        distribution = (
            distribution / total
        )

        if not self.is_stationary(
            distribution
        ):
            raise OperatorError(
                "computed distribution does not satisfy stationarity."
            )

        return readonly_array(
            distribution,
            name="stationary distribution",
            ndim=1,
        )

    def stationarity_residual(
        self,
        distribution,
    ) -> float:
        """
        Return ||A pi||_2 for the stationary system.
        """

        values = _validate_distribution(
            distribution,
            dimension=self.dimension,
            tol=self.tol,
        )

        residual = (
            self.stationary_system_matrix()
            @ values
        )

        return float(
            np.linalg.norm(
                residual,
                ord=2,
            )
        )

    def is_stationary(
        self,
        distribution,
    ) -> bool:
        """
        Check whether a distribution is stationary.
        """

        try:
            residual = (
                self.stationarity_residual(
                    distribution
                )
            )
        except (
            DimensionMismatchError,
            OperatorError,
        ):
            return False

        return bool(
            residual
            <= max(
                self.tol,
                1e-12,
            )
        )

    # -----------------------------------------------------------------------
    # Detailed Balance and Reversibility
    # -----------------------------------------------------------------------

    def row_oriented_matrix(
        self,
    ) -> np.ndarray:
        """
        Return a row-oriented transition or rate matrix.

        This provides one convention-independent representation for
        detailed-balance calculations.
        """

        matrix = self.operator.matrix

        if (
            self.operator.is_row_stochastic
            if self.is_discrete_time
            else self.operator.is_row_generator
        ):
            oriented = matrix
        else:
            oriented = matrix.T

        return readonly_array(
            oriented,
            name="row-oriented stochastic matrix",
            ndim=2,
        )

    def detailed_balance_matrix(
        self,
        distribution=None,
    ) -> np.ndarray:
        """
        Return the detailed-balance defect matrix.

        Entry (i, j) is

            pi_i A_ij - pi_j A_ji,

        where A is row-oriented.
        """

        if distribution is None:
            values = (
                self.stationary_distribution()
            )
        else:
            values = _validate_distribution(
                distribution,
                dimension=self.dimension,
                tol=self.tol,
            )

        matrix = self.row_oriented_matrix()

        flow = (
            values[:, None]
            * matrix
        )

        defect = (
            flow
            - flow.T
        )

        return readonly_array(
            defect,
            name="detailed balance matrix",
            ndim=2,
        )

    def detailed_balance_defect(
        self,
        distribution=None,
    ) -> float:
        """
        Return the Frobenius norm of the detailed-balance defect.
        """

        return float(
            np.linalg.norm(
                self.detailed_balance_matrix(
                    distribution
                ),
                ord="fro",
            )
        )

    def is_reversible(
        self,
        distribution=None,
    ) -> bool:
        """
        Return whether detailed balance holds numerically.
        """

        return bool(
            self.detailed_balance_defect(
                distribution
            )
            <= max(
                self.tol,
                1e-12,
            )
        )

    # -----------------------------------------------------------------------
    # Long-Time Comparison
    # -----------------------------------------------------------------------

    def limiting_distribution(
        self,
        initial_distribution,
        *,
        steps: int = 1000,
        time: float = 100.0,
    ) -> np.ndarray:
        """
        Return a large-step or large-time evolved distribution.

        This is a numerical diagnostic and does not guarantee convergence.
        """

        if self.is_discrete_time:
            return self.operator.evolve_distribution(
                initial_distribution,
                steps=steps,
            )

        return self.operator.evolve_distribution(
            initial_distribution,
            t=time,
        )

    def limiting_error(
        self,
        initial_distribution,
        *,
        steps: int = 1000,
        time: float = 100.0,
    ) -> float:
        """
        Compare long-time evolution with one stationary distribution.
        """

        limiting = (
            self.limiting_distribution(
                initial_distribution,
                steps=steps,
                time=time,
            )
        )

        stationary = (
            self.stationary_distribution()
        )

        return float(
            np.linalg.norm(
                limiting - stationary,
                ord=1,
            )
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return stationary and reversibility diagnostics.
        """

        distribution = (
            self.stationary_distribution()
        )

        return {
            "operator": self.operator.name,
            "model_type": (
                "discrete_time"
                if self.is_discrete_time
                else "continuous_time"
            ),
            "dimension": self.dimension,
            "states": self.states,
            "stationary_distribution":
                tuple(
                    distribution.tolist()
                ),
            "stationary_dimension":
                self.stationary_dimension(),
            "unique": self.is_unique(),
            "stationarity_residual":
                self.stationarity_residual(
                    distribution
                ),
            "detailed_balance_defect":
                self.detailed_balance_defect(
                    distribution
                ),
            "reversible":
                self.is_reversible(
                    distribution
                ),
        }
