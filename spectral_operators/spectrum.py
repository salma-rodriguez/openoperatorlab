"""
rh_operator.spectrum
====================

Spectral analysis tools for LinearOperator objects.

This module contains reusable classes and functions for studying
eigenvalues, eigenvectors, singular values, resolvents, spectral
partitions, and spectral diagnostics.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator, OperatorError, NonSquareOperatorError


# ===========================================================================
# Spectral Analyzer
# ===========================================================================

class SpectralAnalyzer:
    """
    Spectral analysis wrapper for LinearOperator objects.
    """

    def __init__(self, operator: LinearOperator):
        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator

    def eigenvalues(self) -> np.ndarray:
        return self.operator.eigenvalues()

    def eigenvectors(self) -> np.ndarray:
        return self.operator.eigenvectors()

    def eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        return self.operator.eigendecomposition()

    def partition(
        self,
        *,
        alpha: float = 0.13,
        ordering: str = "abs",
    ) -> SpectralPartition:
        """
        Partition the operator spectrum into low, bulk, and high regions.
        """

        return SpectralPartition(
            self.sorted_eigenvalues(ordering=ordering),
            alpha=alpha,
            ordering=ordering,
        )

    def resolvent(self) -> ResolventAnalyzer:
        """
        Return a resolvent analyzer for the operator.
        """

        return ResolventAnalyzer(self.operator)

    def statistics(self, *, ordering: str = "real") -> SpectralStatistics:
        """
        Return spectral statistics for the operator.
        """

        return SpectralStatistics(
            self.eigenvalues(),
            ordering=ordering,
        )

    def sorted_eigenvalues(self, ordering: str = "abs") -> np.ndarray:
        """
        Return eigenvalues sorted according to a chosen ordering.

        Parameters
        ----------
        ordering : str
            Sorting rule. Supported values:

            - "abs"      : sort by absolute value
            - "real"     : sort by real part
            - "imag"     : sort by imaginary part
            - "complex"  : lexicographic sort by real then imaginary part

        Returns
        -------
        np.ndarray
            Sorted eigenvalues.
        """

        vals = self.eigenvalues()

        if ordering == "abs":
            idx = np.argsort(np.abs(vals))

        elif ordering == "real":
            idx = np.argsort(vals.real)

        elif ordering == "imag":
            idx = np.argsort(vals.imag)

        elif ordering == "complex":
            idx = np.lexsort((vals.imag, vals.real))

        else:
            raise OperatorError(
                "ordering must be one of: 'abs', 'real', 'imag', 'complex'."
            )

        return vals[idx]

    def svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.operator.svd()


# ===========================================================================
# Spectral Partition
# ===========================================================================

class SpectralPartition:
    """
    Partition eigenvalues into low, bulk, and high spectral regions.

    Parameters
    ----------
    eigenvalues : array_like
        Spectral values to partition.

    alpha : float
        Fraction of eigenvalues placed in each edge partition.

    ordering : str
        Sorting rule. Supported values are "abs", "real", "imag", "complex".
    """

    def __init__(
        self,
        eigenvalues,
        *,
        alpha: float = 0.13,
        ordering: str = "abs",
    ):

        vals = np.asarray(eigenvalues)

        if vals.ndim != 1:
            raise OperatorError("eigenvalues must be one-dimensional.")

        if not (0.0 <= alpha < 0.5):
            raise OperatorError("alpha must satisfy 0 <= alpha < 0.5.")

        self.eigenvalues = vals
        self.alpha = alpha
        self.ordering = ordering

        self.sorted_values = self._sort(vals, ordering)

        k = int(alpha * len(vals))

        self.k = k
        self.low = self.sorted_values[:k]
        self.bulk = self.sorted_values[k:-k] if k > 0 else self.sorted_values
        self.high = self.sorted_values[-k:] if k > 0 else np.array([], dtype=vals.dtype)

    @staticmethod
    def _sort(vals: np.ndarray, ordering: str) -> np.ndarray:
        if ordering == "abs":
            idx = np.argsort(np.abs(vals))

        elif ordering == "real":
            idx = np.argsort(vals.real)

        elif ordering == "imag":
            idx = np.argsort(vals.imag)

        elif ordering == "complex":
            idx = np.lexsort((vals.imag, vals.real))

        else:
            raise OperatorError(
                "ordering must be one of: 'abs', 'real', 'imag', 'complex'."
            )

        return vals[idx]

    def as_tuple(self):
        """
        Return the partition as (low, bulk, high).
        """

        return self.low, self.bulk, self.high

    def sizes(self) -> tuple[int, int, int]:
        """
        Return partition sizes.
        """

        return len(self.low), len(self.bulk), len(self.high)


# ===========================================================================
# Resolvent Analysis
# ===========================================================================

# ===========================================================================
# Resolvent Analysis
# ===========================================================================

class ResolventAnalyzer:
    """
    Resolvent-based spectral diagnostics.

    Studies operators of the form

        R(z) = (A - zI)^(-1).
    """

    def __init__(self, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        if not operator.is_square:
            raise NonSquareOperatorError(
                "Resolvent analysis requires a square operator."
            )

        self.operator = operator


    def matrix(self, z) -> np.ndarray:
        """
        Compute the resolvent matrix R(z) = (A - zI)^(-1).
        """

        A = self.operator.matrix
        I = np.eye(self.operator.rows, dtype=np.result_type(A, z))

        try:
            return np.linalg.inv(A - z * I)

        except np.linalg.LinAlgError as exc:
            raise OperatorError(
                f"Resolvent is singular at z={z!r}."
            ) from exc


    def norm(self, z, kind: str = "spectral") -> float:
        """
        Compute ||R(z)||.
        """

        R = LinearOperator(
            self.matrix(z),
            name=f"R({z})"
        )

        return R.norm(kind)


    def trace(self, z):
        """
        Compute tr(R(z)).
        """

        return np.trace(self.matrix(z))


    def determinant(self, z):
        """
        Compute det(A - zI).
        """

        A = self.operator.matrix
        I = np.eye(self.operator.rows, dtype=np.result_type(A, z))

        return np.linalg.det(A - z * I)


# ===========================================================================
# Spectral Statistics
# ===========================================================================

class SpectralStatistics:
    """
    Spectral statistics such as gaps, spacings, and summary measures.

    Parameters
    ----------
    eigenvalues : array_like
        One-dimensional array of eigenvalues.

    ordering : str
        Ordering used before computing gaps/spacings.
        Supported values are "abs", "real", "imag", and "complex".
    """

    def __init__(
        self,
        eigenvalues,
        *,
        ordering: str = "real",
    ):

        vals = np.asarray(eigenvalues)

        if vals.ndim != 1:
            raise OperatorError("eigenvalues must be one-dimensional.")

        self.eigenvalues = vals
        self.ordering = ordering
        self.sorted_values = SpectralPartition._sort(vals, ordering)


    def gaps(self) -> np.ndarray:
        """
        Return consecutive spectral gaps.

        For complex spectra, gaps are computed after the chosen ordering.
        """

        return np.diff(self.sorted_values)


    def spacings(self) -> np.ndarray:
        """
        Return absolute consecutive spectral spacings.
        """

        return np.abs(self.gaps())


    def normalized_spacings(self) -> np.ndarray:
        """
        Return spacings normalized by their mean.
        """

        s = self.spacings()

        if len(s) == 0:
            return s

        mean = np.mean(s)

        if mean == 0:
            return s

        return s / mean


    def mean_spacing(self) -> float:
        """
        Return the mean absolute spacing.
        """

        s = self.spacings()

        if len(s) == 0:
            return 0.0

        return float(np.mean(s))


    def min_spacing(self) -> float:
        """
        Return the minimum absolute spacing.
        """

        s = self.spacings()

        if len(s) == 0:
            return 0.0

        return float(np.min(s))


    def max_spacing(self) -> float:
        """
        Return the maximum absolute spacing.
        """

        s = self.spacings()

        if len(s) == 0:
            return 0.0

        return float(np.max(s))


    def variance_spacing(self) -> float:
        """
        Return the variance of absolute spacings.
        """

        s = self.spacings()

        if len(s) == 0:
            return 0.0

        return float(np.var(s))


    def summary(self) -> dict:
        """
        Return a dictionary of basic spectral spacing statistics.
        """

        return {
            "count": int(len(self.eigenvalues)),
            "num_spacings": int(max(len(self.eigenvalues) - 1, 0)),
            "ordering": self.ordering,
            "mean_spacing": self.mean_spacing(),
            "min_spacing": self.min_spacing(),
            "max_spacing": self.max_spacing(),
            "variance_spacing": self.variance_spacing(),
        }
