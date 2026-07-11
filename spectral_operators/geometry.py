"""
rh_operator.geometry
====================

Geometric diagnostics for LinearOperator objects.

This module contains tools for studying symmetry, locality,
boundary effects, operator-induced geometry, and related
diagnostics.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator, OperatorError


# ===========================================================================
# Symmetry Diagnostics
# ===========================================================================

# ===========================================================================
# Symmetry Diagnostics
# ===========================================================================

class SymmetryAnalyzer:
    """
    Analyze symmetry, skew-symmetry, Hermitian structure, and related defects.
    """

    def __init__(self, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator


    def symmetric_defect(self, *, relative: bool = False) -> float:
        """
        Measure the defect from symmetry.

        Computes

            ||A - A^T||_F

        or the relative defect

            ||A - A^T||_F / ||A||_F.
        """

        A = self.operator.matrix
        defect = np.linalg.norm(A - A.T, ord="fro")

        if relative:
            denom = self.operator.norm("fro")
            return float(defect / denom) if denom != 0 else 0.0

        return float(defect)


    def skew_defect(self, *, relative: bool = False) -> float:
        """
        Measure the defect from skew-symmetry.

        Computes

            ||A + A^T||_F.
        """

        A = self.operator.matrix
        defect = np.linalg.norm(A + A.T, ord="fro")

        if relative:
            denom = self.operator.norm("fro")
            return float(defect / denom) if denom != 0 else 0.0

        return float(defect)


    def hermitian_defect(self, *, relative: bool = False) -> float:
        """
        Measure the defect from Hermitian/self-adjoint structure.

        Computes

            ||A - A†||_F.
        """

        A = self.operator.matrix
        defect = np.linalg.norm(A - A.conj().T, ord="fro")

        if relative:
            denom = self.operator.norm("fro")
            return float(defect / denom) if denom != 0 else 0.0

        return float(defect)


    def antihermitian_defect(self, *, relative: bool = False) -> float:
        """
        Measure the defect from anti-Hermitian/skew-adjoint structure.

        Computes

            ||A + A†||_F.
        """

        A = self.operator.matrix
        defect = np.linalg.norm(A + A.conj().T, ord="fro")

        if relative:
            denom = self.operator.norm("fro")
            return float(defect / denom) if denom != 0 else 0.0

        return float(defect)


    def summary(self) -> dict:
        """
        Return a summary of basic symmetry diagnostics.
        """

        return {
            "symmetric": self.operator.is_symmetric(),
            "skew": self.operator.is_skew(),
            "hermitian": self.operator.is_hermitian(),
            "antihermitian": self.operator.is_antihermitian(),
            "symmetric_defect": self.symmetric_defect(),
            "skew_defect": self.skew_defect(),
            "hermitian_defect": self.hermitian_defect(),
            "antihermitian_defect": self.antihermitian_defect(),
            "relative_symmetric_defect": self.symmetric_defect(relative=True),
            "relative_skew_defect": self.skew_defect(relative=True),
            "relative_hermitian_defect": self.hermitian_defect(relative=True),
            "relative_antihermitian_defect": self.antihermitian_defect(relative=True),
        }


# ===========================================================================
# Boundary Diagnostics
# ===========================================================================

class BoundaryAnalyzer:
    """
    Analyze boundary effects and boundary-localized mass.

    The boundary is defined as the first and last `width` rows/columns
    of the operator matrix.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        width: int = 1,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        if width < 1:
            raise OperatorError("width must be at least 1.")

        if width * 2 > min(operator.shape):
            raise OperatorError("width is too large for operator shape.")

        self.operator = operator
        self.width = width


    def boundary_mask(self) -> np.ndarray:
        """
        Return a boolean mask selecting boundary rows and columns.
        """

        rows, cols = self.operator.shape
        w = self.width

        mask = np.zeros((rows, cols), dtype=bool)

        mask[:w, :] = True
        mask[-w:, :] = True
        mask[:, :w] = True
        mask[:, -w:] = True

        return mask


    def interior_mask(self) -> np.ndarray:
        """
        Return a boolean mask selecting non-boundary entries.
        """

        return ~self.boundary_mask()


    def boundary_norm(self) -> float:
        """
        Frobenius norm of the boundary-localized entries.
        """

        A = self.operator.matrix
        mask = self.boundary_mask()

        return float(np.linalg.norm(A[mask]))


    def interior_norm(self) -> float:
        """
        Frobenius norm of the interior entries.
        """

        A = self.operator.matrix
        mask = self.interior_mask()

        return float(np.linalg.norm(A[mask]))


    def total_norm(self) -> float:
        """
        Frobenius norm of the full operator.
        """

        return self.operator.norm("fro")


    def boundary_ratio(self) -> float:
        """
        Ratio of boundary norm to total norm.
        """

        total = self.total_norm()

        if total == 0:
            return 0.0

        return float(self.boundary_norm() / total)


    def interior_ratio(self) -> float:
        """
        Ratio of interior norm to total norm.
        """

        total = self.total_norm()

        if total == 0:
            return 0.0

        return float(self.interior_norm() / total)


    def summary(self) -> dict:
        """
        Return boundary/interior diagnostic summary.
        """

        return {
            "width": self.width,
            "boundary_norm": self.boundary_norm(),
            "interior_norm": self.interior_norm(),
            "total_norm": self.total_norm(),
            "boundary_ratio": self.boundary_ratio(),
            "interior_ratio": self.interior_ratio(),
        }


# ===========================================================================
# Locality Diagnostics
# ===========================================================================

class LocalityAnalyzer:
    """
    Analyze matrix locality.

    Locality is measured by how concentrated the operator entries are
    near the main diagonal.
    """

    def __init__(self, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator


    def distance_matrix(self) -> np.ndarray:
        """
        Return |i - j| for every matrix entry A_ij.
        """

        rows, cols = self.operator.shape
        i = np.arange(rows)[:, None]
        j = np.arange(cols)[None, :]

        return np.abs(i - j)


    def band_mask(self, bandwidth: int) -> np.ndarray:
        """
        Return mask selecting entries with |i - j| <= bandwidth.
        """

        if bandwidth < 0:
            raise OperatorError("bandwidth must be nonnegative.")

        return self.distance_matrix() <= bandwidth


    def band_norm(self, bandwidth: int) -> float:
        """
        Frobenius norm of entries inside a diagonal band.
        """

        A = self.operator.matrix
        mask = self.band_mask(bandwidth)

        return float(np.linalg.norm(A[mask]))


    def off_band_norm(self, bandwidth: int) -> float:
        """
        Frobenius norm of entries outside a diagonal band.
        """

        A = self.operator.matrix
        mask = ~self.band_mask(bandwidth)

        return float(np.linalg.norm(A[mask]))


    def locality_ratio(self, bandwidth: int) -> float:
        """
        Ratio of band norm to total Frobenius norm.
        """

        total = self.operator.norm("fro")

        if total == 0:
            return 0.0

        return float(self.band_norm(bandwidth) / total)


    def off_locality_ratio(self, bandwidth: int) -> float:
        """
        Ratio of off-band norm to total Frobenius norm.
        """

        total = self.operator.norm("fro")

        if total == 0:
            return 0.0

        return float(self.off_band_norm(bandwidth) / total)


    def effective_bandwidth(self, threshold: float = 0.95) -> int:
        """
        Smallest bandwidth capturing at least `threshold` of total norm.
        """

        if not (0.0 <= threshold <= 1.0):
            raise OperatorError("threshold must satisfy 0 <= threshold <= 1.")

        rows, cols = self.operator.shape
        max_bw = max(rows, cols)

        for bw in range(max_bw):
            if self.locality_ratio(bw) >= threshold:
                return bw

        return max_bw


    def summary(self, bandwidth: int = 1) -> dict:
        """
        Return locality diagnostic summary.
        """

        return {
            "bandwidth": bandwidth,
            "band_norm": self.band_norm(bandwidth),
            "off_band_norm": self.off_band_norm(bandwidth),
            "locality_ratio": self.locality_ratio(bandwidth),
            "off_locality_ratio": self.off_locality_ratio(bandwidth),
            "effective_bandwidth_95": self.effective_bandwidth(0.95),
        }


# ===========================================================================
# Geometry Diagnostics
# ===========================================================================

class GeometryAnalyzer:
    """
    High-level geometric diagnostics for LinearOperator objects.

    This class combines symmetry, boundary, and locality diagnostics
    into a single interface.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        boundary_width: int = 1,
    ):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.operator = operator
        self.boundary_width = boundary_width

        self.symmetry = SymmetryAnalyzer(operator)
        self.boundary = BoundaryAnalyzer(
            operator,
            width=boundary_width,
        )
        self.locality = LocalityAnalyzer(operator)


    def summary(self, *, bandwidth: int = 1) -> dict:
        """
        Return combined geometry diagnostics.
        """

        return {
            "operator": self.operator.name,
            "shape": self.operator.shape,
            "field": self.operator.field.value,
            "symmetry": self.symmetry.summary(),
            "boundary": self.boundary.summary(),
            "locality": self.locality.summary(bandwidth=bandwidth),
        }


    def defects(self) -> dict:
        """
        Return core defect diagnostics.
        """

        return {
            "symmetric_defect": self.symmetry.symmetric_defect(),
            "skew_defect": self.symmetry.skew_defect(),
            "hermitian_defect": self.symmetry.hermitian_defect(),
            "antihermitian_defect": self.symmetry.antihermitian_defect(),
            "boundary_ratio": self.boundary.boundary_ratio(),
        }


    def ratios(self, *, bandwidth: int = 1) -> dict:
        """
        Return normalized geometry ratios.
        """

        return {
            "relative_symmetric_defect": self.symmetry.symmetric_defect(relative=True),
            "relative_skew_defect": self.symmetry.skew_defect(relative=True),
            "relative_hermitian_defect": self.symmetry.hermitian_defect(relative=True),
            "relative_antihermitian_defect": self.symmetry.antihermitian_defect(relative=True),
            "boundary_ratio": self.boundary.boundary_ratio(),
            "interior_ratio": self.boundary.interior_ratio(),
            "locality_ratio": self.locality.locality_ratio(bandwidth),
            "off_locality_ratio": self.locality.off_locality_ratio(bandwidth),
        }
