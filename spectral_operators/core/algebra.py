"""
spectral_operators.core.algebra
===============================

Core finite-dimensional linear algebra abstractions for OperatorLab.

This module defines immutable matrix-backed linear operators together
with common matrix operations used throughout the spectral_operators
subpackage.

The implementation is intentionally independent of any particular
scientific application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from .base import OperatorBase
from .exceptions import (
    DimensionMismatchError,
    NonSquareOperatorError,
    OperatorError,
    SingularOperatorError,
)
from .utilities import readonly_array


# ===========================================================================
# Scalar Field
# ===========================================================================

class Field(Enum):
    """Underlying scalar field."""

    REAL = "real"
    COMPLEX = "complex"


# ===========================================================================
# Matrix Norms
# ===========================================================================

class Norm(Enum):
    """
    Supported matrix norms.
    """

    FROBENIUS = "frobenius"

    SPECTRAL = "spectral"

    NUCLEAR = "nuclear"


# ===========================================================================
# Linear Operator
# ===========================================================================

@dataclass(frozen=True)
class LinearOperator(OperatorBase):
    """
    Immutable matrix-backed linear operator.

    Parameters
    ----------
    matrix
        Matrix representation of the operator.

    name
        Human-readable operator name.

    metadata
        Optional user-defined metadata.
    """

    matrix: np.ndarray

    name: str = "LinearOperator"

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __post_init__(self):

        matrix = readonly_array(
            self.matrix,
            name="operator matrix",
            ndim=2,
        )

        object.__setattr__(self, "matrix", matrix)

    _NORM_ALIASES: ClassVar[dict[object, Norm]] = {
        Norm.FROBENIUS: Norm.FROBENIUS,
        "frobenius": Norm.FROBENIUS,
        "fro": Norm.FROBENIUS,

        Norm.SPECTRAL: Norm.SPECTRAL,
        "spectral": Norm.SPECTRAL,
        "operator": Norm.SPECTRAL,
        2: Norm.SPECTRAL,
        "2": Norm.SPECTRAL,

        Norm.NUCLEAR: Norm.NUCLEAR,
        "nuc": Norm.NUCLEAR,
        "nuclear": Norm.NUCLEAR,
        "trace_norm": Norm.NUCLEAR,
        "*": Norm.NUCLEAR,
    }

    # ------------------------------------------------------------------
    # Basic Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions."""
        return self.matrix.shape

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def is_square(self) -> bool:
        return self.rows == self.cols

    @property
    def dtype(self):
        return self.matrix.dtype

    @property
    def field(self) -> Field:
        if np.iscomplexobj(self.matrix):
            return Field.COMPLEX
        return Field.REAL

    # ------------------------------------------------------------------
    # Matrix Views
    # ------------------------------------------------------------------

    @property
    def transpose(self) -> "LinearOperator":
        """
        Return the matrix transpose of the operator.
    
        Notes
        -----
        This is a purely algebraic transpose:
            (A^T)_{ij} = A_{ji}
        """
        return LinearOperator(
            matrix=self.matrix.T.copy(),
            name=self.name + "^T",
            metadata={**self.metadata, "operation": "transpose"}
        )

    @property
    def adjoint(self) -> "LinearOperator":
        """
        Return the Hermitian adjoint of the operator.
    
        Notes
        -----
        This is the conjugate transpose:
            A† = (A*)^T
        """
        return LinearOperator(
            matrix=self.matrix.conj().T.copy(),
            name=self.name + "†",
            metadata={**self.metadata, "operation": "adjoint"}
        )

    # ------------------------------------------------------------------
    # Linear Algebra
    # ------------------------------------------------------------------

    # properties
    @property
    def trace(self):
        """
        Trace of the operator.
    
        Returns
        -------
        Number
            Sum of the diagonal entries,
    
                tr(A) = Σ_i a_ii.
    
        Notes
        -----
        The trace is invariant under similarity transformations.
        """

        return np.trace(self.matrix)

    @property
    def det(self):
        """
        Determinant of the operator.
    
        Returns
        -------
        Number
            det(A).
        """
    
        return np.linalg.det(self.matrix)

    @property
    def rank(self) -> int:
        """
        Matrix rank.
    
        Returns
        -------
        int
            Numerical rank computed via singular values.
        """
    
        return int(np.linalg.matrix_rank(self.matrix))

    @property
    def cond(self) -> float:
        """
        Matrix condition number.
    
        Returns
        -------
        float
            Condition number with respect to the spectral norm.
        """
    
        return float(np.linalg.cond(self.matrix))

    # operators
    def inv(self) -> "LinearOperator":
        """
        Return the inverse operator A^{-1}.
        """

        if not self.is_square:
            raise NonSquareOperatorError(
                "Inverse is only defined for square operators."
            )

        try:
            matrix = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:
            raise SingularOperatorError(
                "Operator is singular and cannot be inverted."
            ) from exc

        return LinearOperator(
            matrix=matrix,
            name=f"inv({self.name})",
            metadata={**self.metadata, "operation": "inv"}
        )

    def pinv(self) -> "LinearOperator":
        """
        Return the Moore-Penrose pseudoinverse operator.
        """

        return LinearOperator(
            matrix=np.linalg.pinv(self.matrix),
            name=f"pinv({self.name})",
            metadata={**self.metadata, "operation": "pinv"}
        )

    # ------------------------------------------------------------------
    # Spectral Theory
    # ------------------------------------------------------------------

    def eigenvalues(self) -> np.ndarray:
        """
        Compute the eigenvalues of the operator.

        Returns
        -------
        np.ndarray
            Array of eigenvalues λ satisfying

                A v = λ v.
        """

        if not self.is_square:
            raise NonSquareOperatorError(
                "Eigenvalues are only defined for square operators."
            )

        return np.linalg.eigvals(self.matrix)


    def eigenvectors(self) -> np.ndarray:
        """
        Compute the right eigenvectors of the operator.

        Returns
        -------
        np.ndarray
            Matrix whose columns are right eigenvectors.

        Notes
        -----
        If V = eigenvectors(), then V[:, k] is the eigenvector
        associated with the kth eigenvalue returned by eigenvalues().
        """

        if not self.is_square:
            raise NonSquareOperatorError(
                "Eigenvectors are only defined for square operators."
            )

        _, vectors = np.linalg.eig(self.matrix)

        return vectors


    def eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the eigendecomposition of the operator.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair (eigenvalues, eigenvectors), where eigenvectors are
            stored columnwise.

        Notes
        -----
        For an operator A, this computes

            A ψ_n = λ_n ψ_n.

        The returned eigenvectors matrix V satisfies approximately

            A @ V = V @ diag(λ).
        """

        if not self.is_square:
            raise NonSquareOperatorError(
                "Eigendecomposition is only defined for square operators."
            )

        values, vectors = np.linalg.eig(self.matrix)

        return values, vectors


    def svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the singular value decomposition of the operator.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Triple (U, s, Vh), where

                A = U @ diag(s) @ Vh.

        Notes
        -----
        This uses NumPy's reduced singular value decomposition.
        """

        U, s, Vh = np.linalg.svd(self.matrix, full_matrices=False)

        return U, s, Vh

    # ------------------------------------------------------------------
    # Norms
    # ------------------------------------------------------------------

    def norm(self, kind: Norm | str | int = Norm.FROBENIUS) -> float:
        """
        Compute a matrix norm.
    
        Parameters
        ----------
        kind : Norm, str or int, optional
            Requested matrix norm.
    
            Accepted values include
    
            - Norm.FROBENIUS
            - Norm.SPECTRAL
            - Norm.NUCLEAR
            - "fro"
            - "frobenius"
            - "spectral"
            - "operator"
            - "nuclear"
            - 2
    
        Returns
        -------
        float
            Requested matrix norm.
        """
    
        try:
            kind = self._NORM_ALIASES[kind]
        except KeyError:
            raise ValueError(f"Unsupported norm type: {kind!r}")
    
        if kind is Norm.FROBENIUS:
            return self._frobenius_norm()
    
        if kind is Norm.SPECTRAL:
            return self._spectral_norm()
    
        if kind is Norm.NUCLEAR:
            return self._nuclear_norm()
    
        raise RuntimeError("Unhandled norm type.")

    def _frobenius_norm(self) -> float:
        """
        Compute the Frobenius norm.
    
        Returns
        -------
        float
            Frobenius norm
    
                ||A||_F = sqrt(sum |a_ij|^2).
    
        Notes
        -----
        The Frobenius norm is unitarily invariant and equals the
        Euclidean norm of the matrix viewed as a vector.
        """
    
        return float(np.linalg.norm(self.matrix, ord="fro"))

    def _spectral_norm(self) -> float:
        """
        Compute the spectral (operator) norm.
    
        Returns
        -------
        float
            Largest singular value of the operator.
    
        Notes
        -----
        The spectral norm is
    
            ||A||₂ = σ_max(A),
    
        where σ_max denotes the largest singular value.
        """
    
        return float(np.linalg.norm(self.matrix, ord=2))

    def _nuclear_norm(self) -> float:
        """
        Compute the nuclear (trace) norm.
    
        Returns
        -------
        float
            Sum of the singular values.
    
        Notes
        -----
        The nuclear norm is
    
            ||A||_* = Σ σ_i,
    
        where σ_i are the singular values.
        """
    
        return float(np.linalg.norm(self.matrix, ord="nuc"))

    # ------------------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------------------

    def is_symmetric(self, tol: float = 1e-10) -> bool:
        """
        Check whether the operator is symmetric.

        Returns
        -------
        bool
            True if A ≈ A^T.
        """

        if not self.is_square:
            return False

        return bool(
            np.allclose(
                self.matrix,
                self.matrix.T,
                atol=tol,
                rtol=tol
            )
        )


    def is_skew(self, tol: float = 1e-10) -> bool:
        """
        Check whether the operator is skew-symmetric.

        Returns
        -------
        bool
            True if A ≈ -A^T.
        """

        if not self.is_square:
            return False

        return bool(
            np.allclose(
                self.matrix,
                -self.matrix.T,
                atol=tol,
                rtol=tol
            )
        )


    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """
        Check whether the operator is Hermitian / self-adjoint.

        Returns
        -------
        bool
            True if A ≈ A†.
        """

        if not self.is_square:
            return False

        return bool(
            np.allclose(
                self.matrix,
                self.matrix.conj().T,
                atol=tol,
                rtol=tol
            )
        )


    def is_antihermitian(self, tol: float = 1e-10) -> bool:
        """
        Check whether the operator is anti-Hermitian / skew-adjoint.

        Returns
        -------
        bool
            True if A ≈ -A†.
        """

        if not self.is_square:
            return False

        return bool(
            np.allclose(
                self.matrix,
                -self.matrix.conj().T,
                atol=tol,
                rtol=tol
            )
        )

    # ------------------------------------------------------------------
    # Decompositions
    # ------------------------------------------------------------------

    def symmetric_part(self) -> "LinearOperator":
        """
        Return the symmetric part of the operator.

        Returns
        -------
        LinearOperator
            The operator

                A_sym = 1/2 (A + A^T).
        """

        return LinearOperator(
            matrix=0.5 * (self.matrix + self.matrix.T),
            name=f"sym({self.name})",
            metadata={**self.metadata, "operation": "symmetric_part"}
        )


    def skew_part(self) -> "LinearOperator":
        """
        Return the skew-symmetric part of the operator.

        Returns
        -------
        LinearOperator
            The operator

                A_skew = 1/2 (A - A^T).
        """

        return LinearOperator(
            matrix=0.5 * (self.matrix - self.matrix.T),
            name=f"skew({self.name})",
            metadata={**self.metadata, "operation": "skew_part"}
        )


    def hermitian_part(self) -> "LinearOperator":
        """
        Return the Hermitian/self-adjoint part of the operator.

        Returns
        -------
        LinearOperator
            The operator

                A_H = 1/2 (A + A†).
        """

        return LinearOperator(
            matrix=0.5 * (self.matrix + self.matrix.conj().T),
            name=f"herm({self.name})",
            metadata={**self.metadata, "operation": "hermitian_part"}
        )


    def antihermitian_part(self) -> "LinearOperator":
        """
        Return the anti-Hermitian/skew-adjoint part of the operator.

        Returns
        -------
        LinearOperator
            The operator

                A_A = 1/2 (A - A†).
        """

        return LinearOperator(
            matrix=0.5 * (self.matrix - self.matrix.conj().T),
            name=f"antiherm({self.name})",
            metadata={**self.metadata, "operation": "antihermitian_part"}
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "LinearOperator") -> "LinearOperator":
    
        if self.shape != other.shape:
            raise DimensionMismatchError("Addition requires same shape.")
    
        return LinearOperator(
            matrix=self.matrix + other.matrix,
            name=f"({self.name}+{other.name})",
            metadata={**self.metadata, "operation": "add"}
        )
    
    
    def __sub__(self, other: "LinearOperator") -> "LinearOperator":
    
        if self.shape != other.shape:
            raise DimensionMismatchError("Subtraction requires same shape.")
    
        return LinearOperator(
            matrix=self.matrix - other.matrix,
            name=f"({self.name}-{other.name})",
            metadata={**self.metadata, "operation": "sub"}
        )
    
    
    def __matmul__(self, other: "LinearOperator") -> "LinearOperator":
    
        if self.cols != other.rows:
            raise DimensionMismatchError(
                "Matrix multiplication shape mismatch."
            )
    
        return LinearOperator(
            matrix=self.matrix @ other.matrix,
            name=f"({self.name}@{other.name})",
            metadata={**self.metadata, "operation": "matmul"}
        )
    
    
    def __mul__(self, scalar: float) -> "LinearOperator":
    
        return LinearOperator(
            matrix=self.matrix * scalar,
            name=f"{scalar}*{self.name}",
            metadata={**self.metadata, "operation": "scale"}
        )
    
    
    def __rmul__(self, scalar: float) -> "LinearOperator":
        return self.__mul__(scalar)
    
    
    def __truediv__(self, scalar: float) -> "LinearOperator":
    
        if scalar == 0:
            raise OperatorError("Division by zero scalar.")
    
        return LinearOperator(
            matrix=self.matrix / scalar,
            name=f"{self.name}/{scalar}",
            metadata={**self.metadata, "operation": "scale"}
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LinearOperator("
            f"name={self.name!r}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"field={self.field.value!r})"
        )


    def __str__(self) -> str:
        return (
            f"{self.name}: "
            f"{self.rows}×{self.cols} "
            f"{self.field.value} operator "
            f"[dtype={self.dtype}]"
        )

# ===========================================================================
# Operator Factory
# ===========================================================================

class OperatorFactory:
    """
    Factory methods for constructing common LinearOperator objects.
    """

    @staticmethod
    def identity(n: int, *, dtype=float, name: str = "I") -> LinearOperator:
        """
        Construct the identity operator I_n.
        """
        return LinearOperator(
            matrix=np.eye(n, dtype=dtype),
            name=name,
            metadata={"factory": "identity", "dimension": n}
        )

    @staticmethod
    def zeros(shape, *, dtype=float, name: str = "0") -> LinearOperator:
        """
        Construct the zero operator.
        """
        return LinearOperator(
            matrix=np.zeros(shape, dtype=dtype),
            name=name,
            metadata={"factory": "zeros", "shape": shape}
        )

    @staticmethod
    def ones(shape, *, dtype=float, name: str = "1") -> LinearOperator:
        """
        Construct an operator whose matrix entries are all one.
        """
        return LinearOperator(
            matrix=np.ones(shape, dtype=dtype),
            name=name,
            metadata={"factory": "ones", "shape": shape}
        )

    @staticmethod
    def diagonal(diagonal, *, dtype=None, name: str = "diag") -> LinearOperator:
        """
        Construct a diagonal operator from a one-dimensional array.
        """
        d = np.asarray(diagonal, dtype=dtype)

        if d.ndim != 1:
            raise OperatorError(
                "Diagonal input must be one-dimensional."
            )

        return LinearOperator(
            matrix=np.diag(d),
            name=name,
            metadata={"factory": "diagonal", "dimension": len(d)}
        )

    @staticmethod
    def random(
        shape,
        *,
        dtype=float,
        seed: int | None = None,
        name: str = "random"
    ) -> LinearOperator:
        """
        Construct a random operator.

        For real dtype, entries are sampled from U(0, 1).
        For complex dtype, real and imaginary parts are sampled independently.
        """
        rng = np.random.default_rng(seed)

        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            matrix = rng.random(shape) + 1j * rng.random(shape)
            matrix = matrix.astype(dtype)
        else:
            matrix = rng.random(shape).astype(dtype)

        return LinearOperator(
            matrix=matrix,
            name=name,
            metadata={"factory": "random", "shape": shape, "seed": seed}
        )

    @staticmethod
    def block(blocks, *, name: str = "block") -> LinearOperator:
        """
        Construct a block operator from nested arrays/operators.

        Parameters
        ----------
        blocks
            Nested list of LinearOperator or ndarray blocks.
        """
        raw_blocks = []

        for row in blocks:
            raw_row = []
            for block in row:
                if isinstance(block, LinearOperator):
                    raw_row.append(block.matrix)
                else:
                    raw_row.append(np.asarray(block))
            raw_blocks.append(raw_row)

        return LinearOperator(
            matrix=np.block(raw_blocks),
            name=name,
            metadata={"factory": "block"}
        )

