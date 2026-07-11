"""
spectral_operator.visualization
===============================

Visualization helpers for spectral operator models.

This module provides lightweight data-preparation utilities for plotting
spectra, eigenvalue distributions, operator matrices, diagnostics, and
zeta-correspondence comparisons.

The module intentionally avoids enforcing a plotting backend. Backends such
as Matplotlib, Plotly, or notebook-specific renderers can be added later.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator, OperatorError
from .spectrum import SpectralAnalyzer
from .geometry import GeometryAnalyzer
from .diagnostics import OperatorDiagnostics
from .zeta import ZetaCorrespondence


# ===========================================================================
# Spectrum Visualization Data
# ===========================================================================

class SpectrumData:
    """
    Prepare eigenvalue data for visualization.
    """

    def __init__(self, operator: LinearOperator, *, ordering: str = "abs"):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator
        self.ordering = ordering
        self.eigenvalues = SpectralAnalyzer(operator).sorted_eigenvalues(ordering)

    def complex_points(self) -> np.ndarray:
        return np.column_stack((self.eigenvalues.real, self.eigenvalues.imag))

    def magnitudes(self) -> np.ndarray:
        return np.abs(self.eigenvalues)

    def phases(self) -> np.ndarray:
        return np.angle(self.eigenvalues)

    def as_dict(self) -> dict:
        return {
            "operator": self.operator.name,
            "ordering": self.ordering,
            "real": self.eigenvalues.real.tolist(),
            "imag": self.eigenvalues.imag.tolist(),
            "magnitude": self.magnitudes().tolist(),
            "phase": self.phases().tolist(),
        }


# ===========================================================================
# Matrix Visualization Data
# ===========================================================================

class MatrixData:
    """
    Prepare operator matrix data for visualization.
    """

    def __init__(self, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator
        self.matrix = operator.matrix

    def real(self) -> np.ndarray:
        return self.matrix.real

    def imag(self) -> np.ndarray:
        return self.matrix.imag

    def magnitude(self) -> np.ndarray:
        return np.abs(self.matrix)

    def phase(self) -> np.ndarray:
        return np.angle(self.matrix)

    def as_dict(self) -> dict:
        return {
            "operator": self.operator.name,
            "shape": self.operator.shape,
            "real": self.real().tolist(),
            "imag": self.imag().tolist(),
            "magnitude": self.magnitude().tolist(),
            "phase": self.phase().tolist(),
        }


# ===========================================================================
# Geometry Visualization Data
# ===========================================================================

class GeometryData:
    """
    Prepare geometry diagnostic data for visualization.
    """

    def __init__(self, operator: LinearOperator, *, boundary_width: int = 1):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator
        self.geometry = GeometryAnalyzer(operator, boundary_width=boundary_width)

    def defect_bars(self) -> dict:
        return self.geometry.defects()

    def ratio_bars(self, *, bandwidth: int = 1) -> dict:
        return self.geometry.ratios(bandwidth=bandwidth)

    def as_dict(self, *, bandwidth: int = 1) -> dict:
        return {
            "operator": self.operator.name,
            "defects": self.defect_bars(),
            "ratios": self.ratio_bars(bandwidth=bandwidth),
        }


# ===========================================================================
# Zeta Visualization Data
# ===========================================================================

class ZetaData:
    """
    Prepare zeta-correspondence data for visualization.
    """

    def __init__(self, correspondence: ZetaCorrespondence):

        if not isinstance(correspondence, ZetaCorrespondence):
            raise OperatorError("correspondence must be a ZetaCorrespondence.")

        self.correspondence = correspondence

    def paired_points(self, n: int | None = None) -> np.ndarray:
        spec, gamma = self.correspondence.paired_values(n=n)
        return np.column_stack((spec, gamma))

    def error_series(self, n: int | None = None) -> np.ndarray:
        spec, gamma = self.correspondence.paired_values(n=n)
        return spec - gamma

    def as_dict(self, n: int | None = None) -> dict:
        spec, gamma = self.correspondence.paired_values(n=n)
        error = spec - gamma

        return {
            "count": int(len(spec)),
            "spectral_values": spec.tolist(),
            "zero_ordinates": gamma.tolist(),
            "errors": error.tolist(),
        }


# ===========================================================================
# Visualization Bundle
# ===========================================================================

# ===========================================================================
# Visualization Bundle
# ===========================================================================

class VisualizationBundle:
    """
    High-level visualization data bundle for a LinearOperator.
    """

    def __init__(self, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator

    def spectrum(self, *, ordering: str = "abs") -> SpectrumData:
        return SpectrumData(self.operator, ordering=ordering)

    def matrix(self) -> MatrixData:
        return MatrixData(self.operator)

    def geometry(self, *, boundary_width: int = 1) -> GeometryData:
        return GeometryData(self.operator, boundary_width=boundary_width)

    def diagnostics(self) -> dict:
        return OperatorDiagnostics(self.operator).summary()

    def as_dict(self, *, ordering: str = "abs", bandwidth: int = 1) -> dict:
        return {
            "operator": self.operator.name,
            "spectrum": self.spectrum(ordering=ordering).as_dict(),
            "matrix": self.matrix().as_dict(),
            "geometry": self.geometry().as_dict(bandwidth=bandwidth),
            "diagnostics": self.diagnostics(),
        }
