"""
spectral_operator.diagnostics
=============================

Diagnostic tools for spectral operator models.

This module provides reusable diagnostic wrappers for validating,
summarizing, and comparing operators across algebraic, spectral,
geometric, evolutionary, adelic, and zeta-oriented dimensions.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator, OperatorError
from .spectrum import SpectralAnalyzer
from .geometry import GeometryAnalyzer
from .evolution import EvolutionAnalyzer
from .zeta import HilbertPolyaAnalyzer, SpectralZeta


# ===========================================================================
# Operator Diagnostics
# ===========================================================================

class OperatorDiagnostics:
    """
    General diagnostics for a single LinearOperator.
    """

    def __init__(self, operator: LinearOperator):
        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator

    def algebra_summary(self) -> dict:
        return {
            "name": self.operator.name,
            "shape": self.operator.shape,
            "field": self.operator.field.value,
            "dtype": str(self.operator.dtype),
            "rank": self.operator.rank,
            "trace": self.operator.trace,
            "det": self.operator.det if self.operator.is_square else None,
            "cond": self.operator.cond if self.operator.is_square else None,
            "frobenius_norm": self.operator.norm("fro"),
            "spectral_norm": self.operator.norm("spectral"),
            "nuclear_norm": self.operator.norm("nuc"),
        }

    def spectral_summary(self) -> dict:
        S = SpectralAnalyzer(self.operator)
        stats = S.statistics()

        return {
            "num_eigenvalues": len(S.eigenvalues()) if self.operator.is_square else None,
            "spacing_summary": stats.summary() if self.operator.is_square else None,
        }

    def geometry_summary(self) -> dict:
        return GeometryAnalyzer(self.operator).summary()

    def hilbert_polya_summary(self) -> dict:
        if not self.operator.is_square:
            return {}

        return HilbertPolyaAnalyzer(self.operator).summary()

    def summary(self) -> dict:
        return {
            "algebra": self.algebra_summary(),
            "spectral": self.spectral_summary(),
            "geometry": self.geometry_summary(),
            "hilbert_polya": self.hilbert_polya_summary(),
        }


# ===========================================================================
# Comparative Diagnostics
# ===========================================================================

class ComparativeDiagnostics:
    """
    Diagnostics for comparing two or more operators.
    """

    def __init__(self, operators):

        if len(operators) == 0:
            raise OperatorError("operators cannot be empty.")

        for op in operators:
            if not isinstance(op, LinearOperator):
                raise OperatorError("all inputs must be LinearOperator instances.")

        self.operators = tuple(operators)

    def norm_table(self, kind: str = "fro") -> dict:
        return {
            op.name: op.norm(kind)
            for op in self.operators
        }

    def pairwise_distances(self, kind: str = "fro") -> dict:
        distances = {}

        for i, A in enumerate(self.operators):
            for j, B in enumerate(self.operators):
                if j <= i:
                    continue

                if A.shape != B.shape:
                    distances[(A.name, B.name)] = None
                else:
                    distances[(A.name, B.name)] = (A - B).norm(kind)

        return distances

    def rank_table(self) -> dict:
        return {
            op.name: op.rank
            for op in self.operators
        }

    def summary(self) -> dict:
        return {
            "num_operators": len(self.operators),
            "names": tuple(op.name for op in self.operators),
            "frobenius_norms": self.norm_table("fro"),
            "spectral_norms": self.norm_table("spectral"),
            "ranks": self.rank_table(),
            "pairwise_frobenius_distances": self.pairwise_distances("fro"),
        }


# ===========================================================================
# Stability Diagnostics
# ===========================================================================

class StabilityDiagnostics:
    """
    Diagnostics for numerical and spectral stability.
    """

    def __init__(self, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator

    def condition_number(self) -> float | None:
        if not self.operator.is_square:
            return None

        return self.operator.cond

    def spectral_radius(self) -> float | None:
        if not self.operator.is_square:
            return None

        vals = self.operator.eigenvalues()

        if len(vals) == 0:
            return 0.0

        return float(np.max(np.abs(vals)))

    def real_spectrum_defect(self) -> float | None:
        if not self.operator.is_square:
            return None

        vals = self.operator.eigenvalues()

        return float(np.linalg.norm(vals.imag))

    def normality_defect(self) -> float | None:
        if not self.operator.is_square:
            return None

        A = self.operator.matrix
        defect = A.conj().T @ A - A @ A.conj().T

        return float(np.linalg.norm(defect, ord="fro"))

    def summary(self) -> dict:
        return {
            "condition_number": self.condition_number(),
            "spectral_radius": self.spectral_radius(),
            "real_spectrum_defect": self.real_spectrum_defect(),
            "normality_defect": self.normality_defect(),
        }


# ===========================================================================
# Research Report
# ===========================================================================

class DiagnosticReport:
    """
    High-level diagnostic report for operator experiments.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        include_zeta: bool = False,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator
        self.include_zeta = include_zeta

    def generate(self) -> dict:
        report = {
            "operator": self.operator.name,
            "diagnostics": OperatorDiagnostics(self.operator).summary(),
            "stability": StabilityDiagnostics(self.operator).summary(),
        }

        if self.include_zeta and self.operator.is_square:
            report["spectral_zeta"] = SpectralZeta(self.operator).summary()

        return report
