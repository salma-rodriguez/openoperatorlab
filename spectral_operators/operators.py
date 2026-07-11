"""
rh_operator.operators
=====================

Concrete operator implementations.

All operators inherit from the immutable LinearOperator
defined in algebra.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import (
    DimensionMismatchError,
    NonSquareOperatorError,
    OperatorError,
)
from .core.utilities import (
    as_one_dimensional_array,
    as_two_dimensional_array,
    normalize_l1,
    require_positive_integer,
    require_positive_real,
    require_same_shape,
)


# ===========================================================================
# Finite Difference Operator
# ===========================================================================

class FiniteDifferenceOperator(LinearOperator):
    """
    Finite difference operator on a uniform one-dimensional grid.

    Parameters
    ----------
    N : int
        Number of grid points.

    L : float
        Half-length of the interval [-L, L].

    derivative : int
        Derivative order. Supported values are 1 and 2.

    boundary : str
        Boundary condition. Supported values are "dirichlet" and "periodic".
    """

    def __init__(
        self,
        N: int,
        L: float,
        *,
        derivative: int = 1,
        boundary: str = "dirichlet",
        dtype=float,
        name: str | None = None,
    ):

        if N < 2:
            raise OperatorError("N must be at least 2.")

        if L <= 0:
            raise OperatorError("L must be positive.")

        if derivative not in (1, 2):
            raise OperatorError(
                "Only first and second derivatives are currently supported."
            )

        if boundary not in ("dirichlet", "periodic"):
            raise OperatorError(
                "Boundary must be either 'dirichlet' or 'periodic'."
            )

        dx = 2.0 * L / (N - 1)

        if derivative == 1:
            matrix = self._first_derivative_matrix(N, dx, boundary, dtype)

        else:
            matrix = self._second_derivative_matrix(N, dx, boundary, dtype)

        if name is None:
            name = f"D{derivative}_{boundary}"

        super().__init__(
            matrix=matrix,
            name=name,
            metadata={
                "operator": "finite_difference",
                "N": N,
                "L": L,
                "dx": dx,
                "derivative": derivative,
                "boundary": boundary,
            },
        )

    @staticmethod
    def _first_derivative_matrix(N, dx, boundary, dtype):
        D = np.zeros((N, N), dtype=dtype)

        for i in range(1, N - 1):
            D[i, i - 1] = -0.5 / dx
            D[i, i + 1] = 0.5 / dx

        if boundary == "dirichlet":
            D[0, 0] = -1.0 / dx
            D[0, 1] = 1.0 / dx

            D[-1, -2] = -1.0 / dx
            D[-1, -1] = 1.0 / dx

        elif boundary == "periodic":
            D[0, -1] = -0.5 / dx
            D[0, 1] = 0.5 / dx

            D[-1, -2] = -0.5 / dx
            D[-1, 0] = 0.5 / dx

        return D

    @staticmethod
    def _second_derivative_matrix(N, dx, boundary, dtype):
        D2 = np.zeros((N, N), dtype=dtype)

        for i in range(1, N - 1):
            D2[i, i - 1] = 1.0 / dx**2
            D2[i, i] = -2.0 / dx**2
            D2[i, i + 1] = 1.0 / dx**2

        if boundary == "dirichlet":
            D2[0, 0] = 1.0
            D2[-1, -1] = 1.0

        elif boundary == "periodic":
            D2[0, -1] = 1.0 / dx**2
            D2[0, 0] = -2.0 / dx**2
            D2[0, 1] = 1.0 / dx**2

            D2[-1, -2] = 1.0 / dx**2
            D2[-1, -1] = -2.0 / dx**2
            D2[-1, 0] = 1.0 / dx**2

        return D2


# ===========================================================================
# Weighted Operator
# ===========================================================================

class WeightedOperator(LinearOperator):
    """
    Weighted operator.

    Constructs either

        W A

    or

        A W

    from a base operator A and a weight operator W.

    Parameters
    ----------
    operator : LinearOperator
        Base operator A.

    weight : LinearOperator or array_like
        Weight matrix or one-dimensional weight vector. If one-dimensional,
        it is converted into a diagonal matrix.

    side : str
        Weighting side. Supported values are "left" and "right".
    """

    def __init__(
        self,
        operator: LinearOperator,
        weight,
        *,
        side: str = "left",
        name: str | None = None,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        if side not in ("left", "right"):
            raise OperatorError("side must be either 'left' or 'right'.")

        W = self._as_weight_matrix(weight)

        if side == "left":
            if W.shape[1] != operator.rows:
                raise OperatorError(
                    "Left weight shape must be compatible with operator rows."
                )
            matrix = W @ operator.matrix

        else:
            if operator.cols != W.shape[0]:
                raise OperatorError(
                    "Right weight shape must be compatible with operator columns."
                )
            matrix = operator.matrix @ W

        if name is None:
            name = f"W@{operator.name}" if side == "left" else f"{operator.name}@W"

        super().__init__(
            matrix=matrix,
            name=name,
            metadata={
                "operator": "weighted",
                "base_operator": operator.name,
                "weight_shape": W.shape,
                "side": side,
            },
        )

    @staticmethod
    def _as_weight_matrix(weight) -> np.ndarray:
        """
        Convert a weight input into a matrix.
        """

        if isinstance(weight, LinearOperator):
            W = weight.matrix

        else:
            W = np.asarray(weight)

        if W.ndim == 1:
            W = np.diag(W)

        if W.ndim != 2:
            raise OperatorError(
                "weight must be a matrix, LinearOperator, or one-dimensional vector."
            )

        return W


# ===========================================================================
# Graded Operator
# ===========================================================================

class GradedOperator(LinearOperator):
    """
    Graded / doubled operator.

    Constructs the block operator

        G(A) = [[0, A],
                [A†, 0]]

    from a base operator A.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        name: str | None = None,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        A = operator.matrix
        A_adj = operator.adjoint.matrix

        zero_top = np.zeros((A.shape[0], A_adj.shape[1]), dtype=A.dtype)
        zero_bottom = np.zeros((A_adj.shape[0], A.shape[1]), dtype=A.dtype)

        matrix = np.block([
            [zero_top, A],
            [A_adj, zero_bottom],
        ])

        if name is None:
            name = f"Graded({operator.name})"

        super().__init__(
            matrix=matrix,
            name=name,
            metadata={
                "operator": "graded",
                "base_operator": operator.name,
                "base_shape": operator.shape,
            },
        )


# ===========================================================================
# Hamiltonian Operator
# ===========================================================================

class HamiltonianOperator(LinearOperator):
    """
    Hamiltonian operator.

    A HamiltonianOperator is a LinearOperator intended to represent
    the generator of a spectral problem

        H ψ = λ ψ

    or a time evolution

        exp(-i H t).

    Parameters
    ----------
    operator : LinearOperator
        Base operator.

    enforce_hermitian : bool
        If True, replace the operator by its Hermitian part

            H = 1/2 (A + A†).
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        enforce_hermitian: bool = True,
        name: str | None = None,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError(
                "operator must be a LinearOperator."
            )

        if enforce_hermitian:
            H = operator.hermitian_part()
        else:
            H = operator

        if name is None:
            name = f"H({operator.name})"

        super().__init__(
            matrix=H.matrix,
            name=name,
            metadata={
                "operator": "hamiltonian",
                "base_operator": operator.name,
                "hermitian": enforce_hermitian,
            },
        )


# ===========================================================================
# Adelic Operators
# ===========================================================================

class AdelicOperator(LinearOperator):
    """
    Adelic-style operator assembled from local components.

    Constructs

        A = sum_j w_j A_j

    where each A_j is a local LinearOperator and w_j is a scalar weight.

    This class is intentionally general: the local components may be
    prime-indexed, place-indexed, scale-indexed, or otherwise labeled.
    """

    def __init__(
        self,
        local_operators,
        *,
        weights=None,
        labels=None,
        normalize: bool = True,
        name: str | None = None,
    ):

        if len(local_operators) == 0:
            raise OperatorError(
                "AdelicOperator requires at least one local operator."
            )

        for op in local_operators:
            if not isinstance(op, LinearOperator):
                raise OperatorError(
                    "All local components must be LinearOperator instances."
                )

        shape = local_operators[0].shape

        for op in local_operators:
            if op.shape != shape:
                raise OperatorError(
                    "All local operators must have the same shape."
                )

        m = len(local_operators)

        if weights is None:
            weights = np.ones(m, dtype=float)
        else:
            weights = np.asarray(weights)

        if weights.ndim != 1 or len(weights) != m:
            raise OperatorError(
                "weights must be one-dimensional and match local_operators."
            )

        if normalize:
            total = np.sum(np.abs(weights))
            if total == 0:
                raise OperatorError(
                    "Cannot normalize zero adelic weights."
                )
            weights = weights / total

        matrix = np.zeros(shape, dtype=np.result_type(
            *[op.matrix for op in local_operators],
            weights
        ))

        for w, op in zip(weights, local_operators):
            matrix = matrix + w * op.matrix

        if labels is None:
            labels = list(range(m))

        if len(labels) != m:
            raise OperatorError(
                "labels must match local_operators."
            )

        if name is None:
            name = "AdelicOperator"

        super().__init__(
            matrix=matrix,
            name=name,
            metadata={
                "operator": "adelic",
                "num_local_operators": m,
                "labels": tuple(labels),
                "weights": tuple(weights.tolist()),
                "normalize": normalize,
            },
        )


# ===========================================================================
# Zeta Operator
# ===========================================================================

class ZetaOperator(LinearOperator):
    """
    Zeta-correspondence operator.

    This class wraps a LinearOperator as a zeta-oriented object without
    assuming the operator was constructed adelically.

    The purpose is to provide a semantic boundary between generic
    operator construction and later zeta/RH-specific diagnostics.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        shift=0.0,
        scale=1.0,
        name: str | None = None,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError(
                "operator must be a LinearOperator."
            )

        if scale == 0:
            raise OperatorError("scale must be nonzero.")

        matrix = scale * operator.matrix

        if shift != 0:
            if not operator.is_square:
                raise OperatorError(
                    "shift can only be applied to square operators."
                )
            matrix = matrix + shift * np.eye(
                operator.rows,
                dtype=matrix.dtype
            )

        if name is None:
            name = f"Zeta({operator.name})"

        super().__init__(
            matrix=matrix,
            name=name,
            metadata={
                "operator": "zeta",
                "base_operator": operator.name,
                "base_shape": operator.shape,
                "shift": shift,
                "scale": scale,
            },
        )
