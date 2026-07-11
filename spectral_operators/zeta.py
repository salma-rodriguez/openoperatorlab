"""
spectral_operator.zeta
======================

Zeta-oriented constructions and diagnostics.

This module contains tools for connecting spectral operator data
with zeta-style objects, spectral zeta functions, zero comparisons,
and Hilbert--Pólya-style investigations.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator, OperatorError
from .operators import ZetaOperator
from .spectrum import SpectralAnalyzer


# ===========================================================================
# Spectral Zeta
# ===========================================================================

class SpectralZeta:
    """
    Spectral zeta function associated with operator eigenvalues.

    Computes

        ζ_A(s) = Σ λ_n^{-s}

    over nonzero eigenvalues.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        discard_zeros: bool = True,
        zero_tol: float = 1e-12,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator
        self.discard_zeros = discard_zeros
        self.zero_tol = zero_tol

        vals = SpectralAnalyzer(operator).eigenvalues()

        if discard_zeros:
            vals = vals[np.abs(vals) > zero_tol]

        self.eigenvalues = vals


    def evaluate(self, s):
        """
        Evaluate the spectral zeta function at s.
        """

        if len(self.eigenvalues) == 0:
            return 0.0

        return np.sum(self.eigenvalues ** (-s))


    def values(self, s_values) -> np.ndarray:
        """
        Evaluate spectral zeta over an array of s-values.
        """

        return np.array([
            self.evaluate(s)
            for s in s_values
        ])


    def summary(self) -> dict:
        """
        Return summary information.
        """

        return {
            "operator": self.operator.name,
            "num_eigenvalues": int(len(self.eigenvalues)),
            "discard_zeros": self.discard_zeros,
            "zero_tol": self.zero_tol,
        }


# ===========================================================================
# Zeta Zero Data
# ===========================================================================

class ZetaZeroSet:
    """
    Container for zeta zero ordinates γ_n.

    The corresponding nontrivial zeros are interpreted as

        ρ_n = 1/2 + i γ_n.
    """

    def __init__(self, gammas):

        g = np.asarray(gammas, dtype=float)

        if g.ndim != 1:
            raise OperatorError("gammas must be one-dimensional.")

        self.gammas = np.sort(g)


    @property
    def zeros(self) -> np.ndarray:
        """
        Return complex zeros 1/2 + i γ_n.
        """

        return 0.5 + 1j * self.gammas


    def first(self, n: int) -> np.ndarray:
        """
        Return first n ordinates.
        """

        if n < 0:
            raise OperatorError("n must be nonnegative.")

        return self.gammas[:n]


    def spacings(self) -> np.ndarray:
        """
        Return consecutive zero spacings.
        """

        return np.diff(self.gammas)


    def summary(self) -> dict:
        """
        Return summary information.
        """

        if len(self.gammas) == 0:
            return {
                "count": 0,
                "min_gamma": None,
                "max_gamma": None,
                "mean_spacing": None,
            }

        spacings = self.spacings()

        return {
            "count": int(len(self.gammas)),
            "min_gamma": float(np.min(self.gammas)),
            "max_gamma": float(np.max(self.gammas)),
            "mean_spacing": float(np.mean(spacings)) if len(spacings) else None,
        }


# ===========================================================================
# Zeta Correspondence
# ===========================================================================

class ZetaCorrespondence:
    """
    Compare operator spectra with zeta-style zero ordinates.
    """

    def __init__(
        self,
        operator: LinearOperator,
        zeros: ZetaZeroSet,
        *,
        ordering: str = "abs",
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        if not isinstance(zeros, ZetaZeroSet):
            raise OperatorError("zeros must be a ZetaZeroSet.")

        self.operator = operator
        self.zeros = zeros
        self.ordering = ordering

        self.spectrum = SpectralAnalyzer(operator).sorted_eigenvalues(ordering)


    def compare(self, n: int | None = None) -> dict:
        """
        Compare first n spectral values with first n zero ordinates.
        """

        spec = np.asarray(self.spectrum)

        # For complex spectra, compare imaginary magnitudes by default.
        if np.iscomplexobj(spec):
            spec_values = np.abs(spec.imag)
        else:
            spec_values = np.abs(spec)

        zero_values = self.zeros.gammas

        m = min(len(spec_values), len(zero_values))

        if n is not None:
            if n < 0:
                raise OperatorError("n must be nonnegative.")
            m = min(m, n)

        spec_values = spec_values[:m]
        zero_values = zero_values[:m]

        if m == 0:
            return {
                "count": 0,
                "mean_abs_error": None,
                "max_abs_error": None,
                "rms_error": None,
            }

        error = spec_values - zero_values

        return {
            "count": int(m),
            "mean_abs_error": float(np.mean(np.abs(error))),
            "max_abs_error": float(np.max(np.abs(error))),
            "rms_error": float(np.sqrt(np.mean(error**2))),
        }


    def paired_values(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return paired spectral values and zero ordinates.
        """

        spec = np.asarray(self.spectrum)

        if np.iscomplexobj(spec):
            spec_values = np.abs(spec.imag)
        else:
            spec_values = np.abs(spec)

        zero_values = self.zeros.gammas

        m = min(len(spec_values), len(zero_values))

        if n is not None:
            m = min(m, n)

        return spec_values[:m], zero_values[:m]


# ===========================================================================
# Hilbert--Polya Diagnostics
# ===========================================================================

class HilbertPolyaAnalyzer:
    """
    Diagnostics for Hilbert--Pólya-style spectral interpretations.
    """

    def __init__(self, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator
        self.spectrum = SpectralAnalyzer(operator)


    def is_candidate_self_adjoint(self, tol: float = 1e-10) -> bool:
        """
        Check whether the operator is Hermitian/self-adjoint.
        """

        return self.operator.is_hermitian(tol=tol)


    def real_spectrum_defect(self) -> float:
        """
        Measure imaginary leakage of the spectrum.

        For an ideal self-adjoint finite-dimensional approximation,
        eigenvalues should be real up to numerical tolerance.
        """

        vals = self.spectrum.eigenvalues()

        return float(np.linalg.norm(vals.imag))


    def summary(self) -> dict:
        """
        Return Hilbert--Pólya-style diagnostic summary.
        """

        vals = self.spectrum.eigenvalues()

        return {
            "operator": self.operator.name,
            "shape": self.operator.shape,
            "is_hermitian": self.operator.is_hermitian(),
            "real_spectrum_defect": self.real_spectrum_defect(),
            "num_eigenvalues": int(len(vals)),
        }
