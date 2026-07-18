"""
Finite-state stochastic kernels.

This module provides an immutable representation of a finite stochastic
kernel. A stochastic kernel describes the conditional transition law

    K(i, j) = P(X_{n+1} = j | X_n = i)

under the row convention, or equivalently its transposed representation
under the column convention.

The kernel abstraction is deliberately kept separate from MarkovOperator.
A kernel represents transition probabilities, while a MarkovOperator
represents the corresponding linear action on distributions or observables.

Simulation belongs in ``monte_carlo.py``. Stationary distributions,
ergodicity, hitting behavior, and package-wide numerical diagnostics belong
in their respective modules.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .markov import MarkovOperator


Convention: TypeAlias = Literal["row", "column"]
State: TypeAlias = Hashable


class StochasticKernel:
    """
    Immutable finite-state stochastic kernel.

    Parameters
    ----------
    matrix : array_like
        Two-dimensional nonnegative transition matrix.

        Under ``convention="row"``, each row is a probability distribution:

            sum_j K[i, j] = 1.

        Under ``convention="column"``, each column is a probability
        distribution:

            sum_i K[i, j] = 1.

    states : sequence of hashable objects, optional
        Labels for the state space. If omitted, integer labels

            0, 1, ..., n - 1

        are used.

    convention : {"row", "column"}, default="row"
        Orientation of the stochastic matrix.

    name : str, optional
        Human-readable kernel name.

    metadata : mapping, optional
        Descriptive metadata. A read-only copy is stored.

    tol : float, default=1e-10
        Absolute tolerance used for stochastic validation.

    normalize : bool, default=False
        If true, normalize the stochastic axis before validation.

        Normalization is intended for small floating-point discrepancies
        or nonnegative weights. Zero-mass rows or columns cannot be
        normalized.

    Notes
    -----
    This class currently represents finite kernels on one state space, so
    the matrix must be square. A later representation layer may generalize
    kernels to different domain and codomain spaces or continuous state
    spaces.
    """

    __slots__ = (
        "_matrix",
        "_states",
        "_state_to_index",
        "_convention",
        "_name",
        "_metadata",
        "_tol",
    )

    def __init__(
        self,
        matrix: ArrayLike,
        *,
        states: Sequence[State] | None = None,
        convention: Convention = "row",
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tol: float = 1e-10,
        normalize: bool = False,
    ) -> None:
        convention = self._validate_convention(convention)
        tol = self._validate_tolerance(tol)

        array = self._coerce_matrix(matrix)

        if normalize:
            array = self._normalized_matrix(
                array,
                convention=convention,
                tol=tol,
            )

        self._validate_stochastic_matrix(
            array,
            convention=convention,
            tol=tol,
        )

        state_labels = self._normalize_states(
            states,
            size=array.shape[0],
        )

        frozen_matrix = np.array(array, dtype=float, copy=True)
        frozen_matrix.setflags(write=False)

        self._matrix = frozen_matrix
        self._states = state_labels
        self._state_to_index = MappingProxyType(
            {state: index for index, state in enumerate(state_labels)}
        )
        self._convention = convention
        self._name = str(name) if name is not None else "StochasticKernel"
        self._metadata = MappingProxyType(dict(metadata or {}))
        self._tol = tol

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_convention(convention: str) -> Convention:
        if convention not in {"row", "column"}:
            raise ValueError(
                "convention must be either 'row' or 'column'."
            )

        return convention

    @staticmethod
    def _validate_tolerance(tol: float) -> float:
        if isinstance(tol, (bool, np.bool_)):
            raise TypeError("tol must be a nonnegative finite real number.")

        try:
            value = float(tol)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "tol must be a nonnegative finite real number."
            ) from exc

        if not np.isfinite(value):
            raise ValueError("tol must be finite.")

        if value < 0.0:
            raise ValueError("tol must be nonnegative.")

        return value

    @staticmethod
    def _coerce_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
        try:
            array = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "matrix must be convertible to a real-valued NumPy array."
            ) from exc

        if array.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")

        rows, columns = array.shape

        if rows == 0 or columns == 0:
            raise ValueError("matrix must be nonempty.")

        if rows != columns:
            raise ValueError(
                "A finite-state stochastic kernel must be square."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "matrix entries must all be finite real numbers."
            )

        return np.array(array, dtype=float, copy=True)

    @staticmethod
    def _normalize_states(
        states: Sequence[State] | None,
        *,
        size: int,
    ) -> tuple[State, ...]:
        if states is None:
            return tuple(range(size))

        try:
            labels = tuple(states)
        except TypeError as exc:
            raise TypeError("states must be a finite sequence.") from exc

        if len(labels) != size:
            raise ValueError(
                "The number of state labels must match the kernel size."
            )

        for state in labels:
            try:
                hash(state)
            except TypeError as exc:
                raise TypeError(
                    "Every state label must be hashable."
                ) from exc

        if len(set(labels)) != len(labels):
            raise ValueError("State labels must be unique.")

        return labels

    @classmethod
    def _validate_stochastic_matrix(
        cls,
        matrix: NDArray[np.float64],
        *,
        convention: Convention,
        tol: float,
    ) -> None:
        minimum = float(np.min(matrix))

        if minimum < -tol:
            raise ValueError(
                "A stochastic kernel cannot contain negative "
                f"probabilities; minimum entry is {minimum}."
            )

        # Values in [-tol, 0) are floating-point artifacts and are accepted.
        axis = 1 if convention == "row" else 0
        masses = np.sum(matrix, axis=axis)

        if not np.allclose(
            masses,
            np.ones_like(masses),
            atol=tol,
            rtol=tol,
        ):
            orientation = "rows" if convention == "row" else "columns"
            maximum_error = float(np.max(np.abs(masses - 1.0)))

            raise ValueError(
                f"Kernel {orientation} must sum to one within tolerance; "
                f"maximum error is {maximum_error}."
            )

    @classmethod
    def _normalized_matrix(
        cls,
        matrix: NDArray[np.float64],
        *,
        convention: Convention,
        tol: float,
    ) -> NDArray[np.float64]:
        if np.min(matrix) < -tol:
            raise ValueError(
                "Cannot normalize a matrix containing materially "
                "negative entries."
            )

        result = np.array(matrix, dtype=float, copy=True)

        # Clip tiny negative roundoff before normalization.
        result[(result < 0.0) & (result >= -tol)] = 0.0

        axis = 1 if convention == "row" else 0
        masses = np.sum(result, axis=axis)

        if np.any(masses <= tol):
            orientation = "row" if convention == "row" else "column"
            raise ValueError(
                f"Cannot normalize a zero-mass {orientation}."
            )

        if convention == "row":
            result = result / masses[:, np.newaxis]
        else:
            result = result / masses[np.newaxis, :]

        return result

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> NDArray[np.float64]:
        """
        Read-only stochastic matrix.
        """

        return self._matrix

    @property
    def shape(self) -> tuple[int, int]:
        """
        Matrix shape.
        """

        return self._matrix.shape

    @property
    def size(self) -> int:
        """
        Number of states.
        """

        return self._matrix.shape[0]

    @property
    def states(self) -> tuple[State, ...]:
        """
        Ordered state labels.
        """

        return self._states

    @property
    def state_to_index(self) -> Mapping[State, int]:
        """
        Read-only mapping from state labels to integer indices.
        """

        return self._state_to_index

    @property
    def convention(self) -> Convention:
        """
        Stochastic orientation.
        """

        return self._convention

    @property
    def name(self) -> str:
        """
        Human-readable kernel name.
        """

        return self._name

    @property
    def metadata(self) -> Mapping[str, Any]:
        """
        Read-only metadata.
        """

        return self._metadata

    @property
    def tol(self) -> float:
        """
        Validation tolerance.
        """

        return self._tol

    @property
    def is_square(self) -> bool:
        """
        Whether the finite representation is square.

        This is always true for the current kernel implementation.
        """

        return True

    @property
    def is_finite(self) -> bool:
        """
        Whether this is a finite-state kernel.

        This is always true for the current implementation.
        """

        return True

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------

    def index_of(self, state: State) -> int:
        """
        Return the integer index of a state label.

        Parameters
        ----------
        state : hashable
            State label.

        Returns
        -------
        int
            State index.

        Raises
        ------
        KeyError
            If the state does not belong to this kernel.
        """

        try:
            return self._state_to_index[state]
        except KeyError as exc:
            raise KeyError(
                f"Unknown state {state!r}."
            ) from exc

    def state_at(self, index: int) -> State:
        """
        Return the state label at an integer index.
        """

        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError("index must be an integer.")

        index = int(index)

        if index < 0 or index >= self.size:
            raise IndexError(
                f"State index {index} is outside [0, {self.size})."
            )

        return self._states[index]

    # ------------------------------------------------------------------
    # Orientation and distributions
    # ------------------------------------------------------------------

    def row_matrix(self) -> NDArray[np.float64]:
        """
        Return a read-only row-stochastic representation.

        Returns
        -------
        numpy.ndarray
            Matrix ``P`` satisfying ``P.sum(axis=1) == 1``.
        """

        if self._convention == "row":
            return self._matrix

        result = np.array(self._matrix.T, copy=True)
        result.setflags(write=False)
        return result

    def column_matrix(self) -> NDArray[np.float64]:
        """
        Return a read-only column-stochastic representation.

        Returns
        -------
        numpy.ndarray
            Matrix ``P`` satisfying ``P.sum(axis=0) == 1``.
        """

        if self._convention == "column":
            return self._matrix

        result = np.array(self._matrix.T, copy=True)
        result.setflags(write=False)
        return result

    def distribution(
        self,
        state: State,
        *,
        labeled: bool = False,
    ) -> NDArray[np.float64] | Mapping[State, float]:
        """
        Return the one-step transition distribution from a state.

        Parameters
        ----------
        state : hashable
            Starting state.

        labeled : bool, default=False
            If true, return a read-only state-to-probability mapping.
            Otherwise return a read-only NumPy vector.

        Returns
        -------
        numpy.ndarray or mapping
            One-step transition probabilities.
        """

        index = self.index_of(state)
        probabilities = np.array(
            self.row_matrix()[index],
            dtype=float,
            copy=True,
        )
        probabilities.setflags(write=False)

        if not labeled:
            return probabilities

        return MappingProxyType(
            {
                target: float(probabilities[j])
                for j, target in enumerate(self._states)
            }
        )

    def probability(
        self,
        source: State,
        target: State,
    ) -> float:
        """
        Return ``K(source, target)``.
        """

        source_index = self.index_of(source)
        target_index = self.index_of(target)

        return float(
            self.row_matrix()[source_index, target_index]
        )

    def support(
        self,
        state: State,
        *,
        tol: float | None = None,
    ) -> tuple[State, ...]:
        """
        Return states having positive transition probability.

        Parameters
        ----------
        state : hashable
            Starting state.

        tol : float, optional
            Support threshold. The kernel tolerance is used by default.
        """

        threshold = (
            self._tol
            if tol is None
            else self._validate_tolerance(tol)
        )

        probabilities = self.distribution(state)

        return tuple(
            target
            for target, probability in zip(
                self._states,
                probabilities,
                strict=True,
            )
            if probability > threshold
        )

    # ------------------------------------------------------------------
    # Stochastic checks
    # ------------------------------------------------------------------

    def stochastic_sums(self) -> NDArray[np.float64]:
        """
        Return row or column sums according to the stored convention.
        """

        axis = 1 if self._convention == "row" else 0
        result = np.sum(self._matrix, axis=axis)
        result.setflags(write=False)
        return result

    def row_sums(self) -> NDArray[np.float64]:
        """
        Return sums of the row-oriented transition matrix.
        """

        result = np.sum(self.row_matrix(), axis=1)
        result.setflags(write=False)
        return result

    def column_sums(self) -> NDArray[np.float64]:
        """
        Return sums of the column-oriented transition matrix.
        """

        result = np.sum(self.column_matrix(), axis=0)
        result.setflags(write=False)
        return result

    def is_normalized(self, tol: float | None = None) -> bool:
        """
        Check whether all stochastic-axis sums are approximately one.
        """

        threshold = (
            self._tol
            if tol is None
            else self._validate_tolerance(tol)
        )

        sums = self.stochastic_sums()

        return bool(
            np.allclose(
                sums,
                np.ones_like(sums),
                atol=threshold,
                rtol=threshold,
            )
        )

    def is_deterministic(self, tol: float | None = None) -> bool:
        """
        Check whether every state has a deterministic successor.

        A finite kernel is deterministic when every row-oriented
        transition distribution contains exactly one probability near one
        and all remaining probabilities near zero.
        """

        threshold = (
            self._tol
            if tol is None
            else self._validate_tolerance(tol)
        )

        matrix = self.row_matrix()
        maxima = np.max(matrix, axis=1)
        positive_counts = np.sum(matrix > threshold, axis=1)

        return bool(
            np.allclose(
                maxima,
                np.ones_like(maxima),
                atol=threshold,
                rtol=threshold,
            )
            and np.all(positive_counts == 1)
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(
        cls,
        size: int | None = None,
        *,
        states: Sequence[State] | None = None,
        convention: Convention = "row",
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tol: float = 1e-10,
    ) -> StochasticKernel:
        """
        Construct the identity kernel.

        Every state transitions to itself with probability one.
        """

        resolved_size, resolved_states = cls._resolve_size_and_states(
            size=size,
            states=states,
        )

        return cls(
            np.eye(resolved_size, dtype=float),
            states=resolved_states,
            convention=convention,
            name=name or "IdentityKernel",
            metadata={
                **dict(metadata or {}),
                "kernel_type": "identity",
            },
            tol=tol,
        )

    @classmethod
    def uniform(
        cls,
        size: int | None = None,
        *,
        states: Sequence[State] | None = None,
        convention: Convention = "row",
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tol: float = 1e-10,
    ) -> StochasticKernel:
        """
        Construct a uniform kernel.

        Every state transitions uniformly to every state.
        """

        resolved_size, resolved_states = cls._resolve_size_and_states(
            size=size,
            states=states,
        )

        matrix = np.full(
            (resolved_size, resolved_size),
            1.0 / resolved_size,
            dtype=float,
        )

        return cls(
            matrix,
            states=resolved_states,
            convention=convention,
            name=name or "UniformKernel",
            metadata={
                **dict(metadata or {}),
                "kernel_type": "uniform",
            },
            tol=tol,
        )

    @classmethod
    def absorbing(
        cls,
        absorbing_states: State | Iterable[State],
        *,
        states: Sequence[State],
        convention: Convention = "row",
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tol: float = 1e-10,
    ) -> StochasticKernel:
        """
        Construct a simple absorbing kernel.

        Absorbing states remain fixed. Every nonabsorbing state transitions
        uniformly among the absorbing states.

        Parameters
        ----------
        absorbing_states : state or iterable of states
            Nonempty set of absorbing states.

        states : sequence
            Full state space.
        """

        labels = tuple(states)

        if not labels:
            raise ValueError("states must be nonempty.")

        if isinstance(absorbing_states, (str, bytes)):
            targets = (absorbing_states,)
        else:
            try:
                targets = tuple(absorbing_states)
            except TypeError:
                targets = (absorbing_states,)

        if not targets:
            raise ValueError(
                "At least one absorbing state is required."
            )

        if len(set(targets)) != len(targets):
            raise ValueError(
                "absorbing_states cannot contain duplicates."
            )

        state_to_index = {
            state: index for index, state in enumerate(labels)
        }

        unknown = [
            state for state in targets
            if state not in state_to_index
        ]

        if unknown:
            raise KeyError(
                f"Unknown absorbing states: {unknown!r}."
            )

        matrix = np.zeros((len(labels), len(labels)), dtype=float)
        target_indices = [state_to_index[state] for state in targets]

        for index in range(len(labels)):
            if index in target_indices:
                matrix[index, index] = 1.0
            else:
                matrix[index, target_indices] = (
                    1.0 / len(target_indices)
                )

        if convention == "column":
            matrix = matrix.T

        return cls(
            matrix,
            states=labels,
            convention=convention,
            name=name or "AbsorbingKernel",
            metadata={
                **dict(metadata or {}),
                "kernel_type": "absorbing",
                "absorbing_states": targets,
            },
            tol=tol,
        )

    @staticmethod
    def _resolve_size_and_states(
        *,
        size: int | None,
        states: Sequence[State] | None,
    ) -> tuple[int, tuple[State, ...]]:
        if size is None and states is None:
            raise ValueError(
                "Either size or states must be provided."
            )

        labels: tuple[State, ...]

        if states is None:
            if isinstance(size, (bool, np.bool_)) or not isinstance(
                size,
                (int, np.integer),
            ):
                raise TypeError("size must be a positive integer.")

            resolved_size = int(size)

            if resolved_size <= 0:
                raise ValueError("size must be positive.")

            labels = tuple(range(resolved_size))
            return resolved_size, labels

        labels = tuple(states)

        if not labels:
            raise ValueError("states must be nonempty.")

        if size is not None:
            if isinstance(size, (bool, np.bool_)) or not isinstance(
                size,
                (int, np.integer),
            ):
                raise TypeError("size must be a positive integer.")

            if int(size) != len(labels):
                raise ValueError(
                    "size must match the number of states."
                )

        return len(labels), labels

    @classmethod
    def from_operator(
        cls,
        operator: MarkovOperator,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tol: float | None = None,
    ) -> StochasticKernel:
        """
        Construct a kernel from a Markov operator.
        """

        if not isinstance(operator, MarkovOperator):
            raise TypeError(
                "operator must be a MarkovOperator."
            )

        convention = getattr(operator, "convention", "row")
        states = getattr(
            operator,
            "states",
            tuple(range(operator.matrix.shape[0])),
        )
        operator_tol = getattr(operator, "tol", 1e-10)

        combined_metadata = dict(
            getattr(operator, "metadata", {}) or {}
        )
        combined_metadata.update(metadata or {})
        combined_metadata["source"] = "MarkovOperator"

        return cls(
            operator.matrix,
            states=states,
            convention=convention,
            name=name or f"kernel({operator.name})",
            metadata=combined_metadata,
            tol=operator_tol if tol is None else tol,
        )

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def normalize(
        self,
        *,
        tol: float | None = None,
    ) -> StochasticKernel:
        """
        Return a normalized copy of the kernel.

        This is mainly useful after constructing a kernel from nonnegative
        weights outside the validated constructor.
        """

        threshold = (
            self._tol
            if tol is None
            else self._validate_tolerance(tol)
        )

        matrix = self._normalized_matrix(
            self._matrix,
            convention=self._convention,
            tol=threshold,
        )

        return type(self)(
            matrix,
            states=self._states,
            convention=self._convention,
            name=f"normalize({self._name})",
            metadata={
                **dict(self._metadata),
                "operation": "normalize",
            },
            tol=threshold,
        )

    def with_convention(
        self,
        convention: Convention,
    ) -> StochasticKernel:
        """
        Return an equivalent kernel in another matrix convention.
        """

        convention = self._validate_convention(convention)

        if convention == self._convention:
            return self.copy()

        matrix = self._matrix.T

        return type(self)(
            matrix,
            states=self._states,
            convention=convention,
            name=self._name,
            metadata={
                **dict(self._metadata),
                "operation": "change_convention",
                "previous_convention": self._convention,
            },
            tol=self._tol,
        )

    def relabel(
        self,
        states: Sequence[State],
        *,
        name: str | None = None,
    ) -> StochasticKernel:
        """
        Return a copy with new state labels.
        """

        labels = self._normalize_states(
            states,
            size=self.size,
        )

        return type(self)(
            self._matrix,
            states=labels,
            convention=self._convention,
            name=name or self._name,
            metadata={
                **dict(self._metadata),
                "operation": "relabel",
                "previous_states": self._states,
            },
            tol=self._tol,
        )

    def copy(
        self,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StochasticKernel:
        """
        Return an independent immutable copy.
        """

        combined_metadata = dict(self._metadata)
        combined_metadata.update(metadata or {})

        return type(self)(
            self._matrix,
            states=self._states,
            convention=self._convention,
            name=name or self._name,
            metadata=combined_metadata,
            tol=self._tol,
        )

    # ------------------------------------------------------------------
    # Composition and powers
    # ------------------------------------------------------------------

    def compose(
        self,
        other: StochasticKernel,
        *,
        name: str | None = None,
    ) -> StochasticKernel:
        """
        Compose this kernel with another kernel.

        ``self.compose(other)`` means: apply ``other`` first and then
        apply ``self``.

        Under the row-oriented representation, the resulting transition
        matrix is

            P_other @ P_self.

        Parameters
        ----------
        other : StochasticKernel
            Kernel applied first.

        Returns
        -------
        StochasticKernel
            Composite kernel.

        Raises
        ------
        TypeError
            If ``other`` is not a stochastic kernel.

        ValueError
            If the state spaces differ.
        """

        if not isinstance(other, StochasticKernel):
            raise TypeError(
                "other must be a StochasticKernel."
            )

        if self._states != other._states:
            raise ValueError(
                "Kernel composition requires identical ordered "
                "state spaces."
            )

        row_matrix = other.row_matrix() @ self.row_matrix()

        result = StochasticKernel(
            row_matrix,
            states=self._states,
            convention="row",
            name=name or f"({self._name}∘{other._name})",
            metadata={
                "operation": "composition",
                "outer_kernel": self._name,
                "inner_kernel": other._name,
            },
            tol=max(self._tol, other._tol),
        )

        return result.with_convention(self._convention)

    def then(
        self,
        other: StochasticKernel,
        *,
        name: str | None = None,
    ) -> StochasticKernel:
        """
        Apply this kernel first and ``other`` second.

        This is equivalent to

            other.compose(self).
        """

        if not isinstance(other, StochasticKernel):
            raise TypeError(
                "other must be a StochasticKernel."
            )

        return other.compose(self, name=name)

    def power(self, exponent: int) -> StochasticKernel:
        """
        Return the n-step kernel.

        Parameters
        ----------
        exponent : int
            Nonnegative integer power.

        Returns
        -------
        StochasticKernel
            Kernel representing ``exponent`` transition steps.
        """

        if isinstance(exponent, (bool, np.bool_)) or not isinstance(
            exponent,
            (int, np.integer),
        ):
            raise TypeError(
                "exponent must be a nonnegative integer."
            )

        exponent = int(exponent)

        if exponent < 0:
            raise ValueError(
                "exponent must be nonnegative."
            )

        row_matrix = np.linalg.matrix_power(
            self.row_matrix(),
            exponent,
        )

        result = type(self)(
            row_matrix,
            states=self._states,
            convention="row",
            name=f"{self._name}^{exponent}",
            metadata={
                **dict(self._metadata),
                "operation": "power",
                "exponent": exponent,
            },
            tol=self._tol,
        )

        return result.with_convention(self._convention)

    def __matmul__(
        self,
        other: StochasticKernel,
    ) -> StochasticKernel:
        """
        Compose kernels using ``@``.

        ``self @ other`` applies ``other`` first and ``self`` second.
        """

        return self.compose(other)

    def __pow__(
        self,
        exponent: int,
    ) -> StochasticKernel:
        return self.power(exponent)

    # ------------------------------------------------------------------
    # Markov conversion
    # ------------------------------------------------------------------

    def to_operator(
        self,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MarkovOperator:
        """
        Convert this kernel to a Markov operator.

        Notes
        -----
        This assumes that ``MarkovOperator`` accepts the package's standard
        constructor fields ``matrix``, ``convention``, ``states``, ``name``,
        ``metadata``, and ``tol``. Adjust only this adapter if the local
        MarkovOperator constructor uses different parameter names.
        """

        combined_metadata = dict(self._metadata)
        combined_metadata.update(metadata or {})
        combined_metadata["source"] = "StochasticKernel"

        return MarkovOperator(
            matrix=self._matrix,
            convention=self._convention,
            states=self._states,
            name=name or f"operator({self._name})",
            metadata=combined_metadata,
            tol=self._tol,
        )

    def to_matrix(
        self,
        *,
        copy: bool = True,
    ) -> NDArray[np.float64]:
        """
        Return the matrix representation.

        Parameters
        ----------
        copy : bool, default=True
            If true, return a writable independent copy. If false, return
            the internal read-only array.
        """

        if not copy:
            return self._matrix

        return np.array(self._matrix, copy=True)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Return a compact descriptive summary.
        """

        return {
            "name": self._name,
            "shape": self.shape,
            "size": self.size,
            "states": self._states,
            "convention": self._convention,
            "is_normalized": self.is_normalized(),
            "is_deterministic": self.is_deterministic(),
            "minimum_probability": float(np.min(self._matrix)),
            "maximum_probability": float(np.max(self._matrix)),
            "metadata": dict(self._metadata),
        }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self._name!r}, "
            f"size={self.size}, "
            f"convention={self._convention!r})"
        )


# Concise public alias. The explicit class name remains available for
# mathematical clarity and future differentiation from continuous kernels.
Kernel = StochasticKernel


__all__ = [
    "Convention",
    "Kernel",
    "State",
    "StochasticKernel",
]
