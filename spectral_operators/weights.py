"""
rh_operator.weights
===================

Weight constructions for operator-theoretic models.

This module contains reusable weight objects used to construct
weighted operators, position-dependent operators, adelic weights,
and related weighting schemes.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator, OperatorError


# ===========================================================================
# Base Weight
# ===========================================================================

class WeightOperator(LinearOperator):
    """
    Base diagonal weight operator.

    Parameters
    ----------
    weights : array_like
        One-dimensional weight values.

    name : str, optional
        Operator name.
    """

    def __init__(
        self,
        weights,
        *,
        dtype=None,
        name: str = "WeightOperator",
    ):

        w = np.asarray(weights, dtype=dtype)

        if w.ndim != 1:
            raise OperatorError("weights must be one-dimensional.")

        super().__init__(
            matrix=np.diag(w),
            name=name,
            metadata={
                "operator": "weight",
                "dimension": len(w),
            },
        )

        object.__setattr__(self, "weights", w)


# ===========================================================================
# Position Weights
# ===========================================================================

class PositionWeight(WeightOperator):
    """
    Position-dependent diagonal weight operator.

    Constructs weights on a uniform grid over [-L, L].

    Parameters
    ----------
    N : int
        Number of grid points.

    L : float
        Half-length of interval [-L, L].

    power : float
        Power used in the weight profile.

    scale : float
        Scaling coefficient.

    offset : float
        Base offset.

    normalize : bool
        If True, divide weights by their maximum absolute value.
    """

    def __init__(
        self,
        N: int,
        L: float,
        *,
        power: float = 2.0,
        scale: float = 1.0,
        offset: float = 1.0,
        normalize: bool = False,
        dtype=float,
        name: str | None = None,
    ):

        if N < 1:
            raise OperatorError("N must be positive.")

        if L <= 0:
            raise OperatorError("L must be positive.")

        x = np.linspace(-L, L, N, dtype=dtype)

        weights = offset + scale * np.abs(x) ** power

        if normalize:
            max_abs = np.max(np.abs(weights))

            if max_abs == 0:
                raise OperatorError("Cannot normalize zero position weights.")

            weights = weights / max_abs

        if name is None:
            name = "PositionWeight"

        super().__init__(
            weights,
            dtype=dtype,
            name=name,
        )

        object.__setattr__(self, "grid", x)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "normalize", normalize)


# ===========================================================================
# Polynomial Weights
# ===========================================================================

class PolynomialWeight(WeightOperator):
    """
    Polynomial diagonal weight operator.

    Constructs

        w(x) = c0 + c1 x + c2 x^2 + ...

    on a uniform grid over [-L, L].
    """

    def __init__(
        self,
        N: int,
        L: float,
        coefficients,
        *,
        normalize: bool = False,
        dtype=float,
        name: str | None = None,
    ):

        if N < 1:
            raise OperatorError("N must be positive.")

        if L <= 0:
            raise OperatorError("L must be positive.")

        coeffs = np.asarray(coefficients, dtype=dtype)

        if coeffs.ndim != 1:
            raise OperatorError("coefficients must be one-dimensional.")

        x = np.linspace(-L, L, N, dtype=dtype)

        weights = np.zeros(N, dtype=dtype)

        for k, c in enumerate(coeffs):
            weights = weights + c * x**k

        if normalize:
            max_abs = np.max(np.abs(weights))

            if max_abs == 0:
                raise OperatorError("Cannot normalize zero polynomial weights.")

            weights = weights / max_abs

        if name is None:
            name = "PolynomialWeight"

        super().__init__(
            weights,
            dtype=dtype,
            name=name,
        )

        object.__setattr__(self, "grid", x)
        object.__setattr__(self, "coefficients", coeffs)
        object.__setattr__(self, "normalize", normalize)


# ===========================================================================
# Exponential Weights
# ===========================================================================

class ExponentialWeight(WeightOperator):
    """
    Exponential diagonal weight operator.

    Constructs

        w(x) = offset + scale * exp(rate * |x|^power)

    on a uniform grid over [-L, L].
    """

    def __init__(
        self,
        N: int,
        L: float,
        *,
        rate: float = -1.0,
        power: float = 2.0,
        scale: float = 1.0,
        offset: float = 0.0,
        normalize: bool = False,
        dtype=float,
        name: str | None = None,
    ):

        if N < 1:
            raise OperatorError("N must be positive.")

        if L <= 0:
            raise OperatorError("L must be positive.")

        x = np.linspace(-L, L, N, dtype=dtype)

        weights = offset + scale * np.exp(rate * np.abs(x) ** power)

        if normalize:
            max_abs = np.max(np.abs(weights))

            if max_abs == 0:
                raise OperatorError("Cannot normalize zero exponential weights.")

            weights = weights / max_abs

        if name is None:
            name = "ExponentialWeight"

        super().__init__(
            weights,
            dtype=dtype,
            name=name,
        )

        object.__setattr__(self, "grid", x)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "normalize", normalize)


# ===========================================================================
# Prime / Adelic Weights
# ===========================================================================

# ===========================================================================
# Prime / Adelic Weights
# ===========================================================================

class PrimeWeight:
    """
    Scalar weights associated with prime-indexed local operators.
    """

    def __init__(
        self,
        primes,
        *,
        rule: str = "inverse",
        normalize: bool = True,
    ):

        p = np.asarray(primes, dtype=float)

        if p.ndim != 1:
            raise OperatorError("primes must be one-dimensional.")

        if np.any(p <= 1):
            raise OperatorError("prime values must be greater than 1.")

        if rule == "inverse":
            weights = 1.0 / p

        elif rule == "log_inverse":
            weights = 1.0 / np.log(p)

        elif rule == "log":
            weights = np.log(p)

        elif rule == "uniform":
            weights = np.ones_like(p)

        else:
            raise OperatorError(
                "rule must be one of: 'inverse', 'log_inverse', 'log', 'uniform'."
            )

        if normalize:
            total = np.sum(np.abs(weights))

            if total == 0:
                raise OperatorError("Cannot normalize zero prime weights.")

            weights = weights / total

        self.primes = p
        self.rule = rule
        self.normalize = normalize
        self.weights = weights

    def as_array(self) -> np.ndarray:
        return self.weights.copy()

    def as_dict(self) -> dict:
        return {
            int(p): float(w)
            for p, w in zip(self.primes, self.weights)
        }


class AdelicWeight:
    """
    Weight system for assembling local or adelic operator components.
    """

    def __init__(
        self,
        labels,
        weights=None,
        *,
        normalize: bool = True,
    ):

        labels = tuple(labels)

        if len(labels) == 0:
            raise OperatorError("labels cannot be empty.")

        if weights is None:
            w = np.ones(len(labels), dtype=float)
        else:
            w = np.asarray(weights, dtype=float)

        if w.ndim != 1 or len(w) != len(labels):
            raise OperatorError("weights must match labels.")

        if normalize:
            total = np.sum(np.abs(w))

            if total == 0:
                raise OperatorError("Cannot normalize zero adelic weights.")

            w = w / total

        self.labels = labels
        self.normalize = normalize
        self.weights = w

    @classmethod
    def from_primes(
        cls,
        primes,
        *,
        rule: str = "inverse",
        normalize: bool = True,
    ) -> "AdelicWeight":

        pw = PrimeWeight(
            primes,
            rule=rule,
            normalize=normalize,
        )

        return cls(
            labels=tuple(int(p) for p in pw.primes),
            weights=pw.weights,
            normalize=False,
        )

    def as_array(self) -> np.ndarray:
        return self.weights.copy()

    def as_dict(self) -> dict:
        return {
            label: float(weight)
            for label, weight in zip(self.labels, self.weights)
        }
