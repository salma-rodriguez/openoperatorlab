"""
Monte Carlo simulation and empirical estimation.

This module provides the stochastic simulation layer for
``stochastic_operators``.

The implementation is developed incrementally:

1. Immutable Monte Carlo result objects.
2. Primitive path simulation.
3. Empirical estimators.
4. Statistical summaries and confidence intervals.

The current implementation defines the result layer used by all later
Monte Carlo routines.

Notes
-----
No package-wide numerical constants are introduced here yet. Defaults and
diagnostic tolerances will be reviewed after the simulation, estimation,
and diagnostics layers are complete.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ============================================================================
# Type aliases
# ============================================================================

State: TypeAlias = Hashable
Path: TypeAlias = tuple[State, ...]
Paths: TypeAlias = tuple[Path, ...]
ConfidenceInterval: TypeAlias = tuple[float, float]


# ============================================================================
# Internal validation and freezing helpers
# ============================================================================


def _freeze_array(
    value: ArrayLike | None,
    *,
    name: str,
    dtype: np.dtype[Any] | type | None = None,
    ndim: int | None = None,
    allow_empty: bool = True,
) -> NDArray[Any] | None:
    """
    Convert an array-like object into an immutable NumPy array.

    Parameters
    ----------
    value
        Array-like value or ``None``.

    name
        Field name used in validation messages.

    dtype
        Optional NumPy dtype.

    ndim
        Required number of dimensions, if specified.

    allow_empty
        Whether an empty array is permitted.

    Returns
    -------
    numpy.ndarray or None
        Independent read-only array.

    Raises
    ------
    TypeError
        If the value cannot be converted to an array.

    ValueError
        If its dimensionality is invalid or an empty array is forbidden.
    """

    if value is None:
        return None

    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be convertible to a NumPy array."
        ) from exc

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name} must have exactly {ndim} dimension"
            f"{'s' if ndim != 1 else ''}."
        )

    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must be nonempty.")

    frozen = np.array(array, copy=True)
    frozen.setflags(write=False)

    return frozen


def _freeze_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """
    Return a read-only copy of metadata.
    """

    if metadata is None:
        return MappingProxyType({})

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")

    return MappingProxyType(dict(metadata))


def _freeze_path(
    path: Sequence[State],
    *,
    name: str,
    allow_empty: bool = True,
) -> Path:
    """
    Convert a state path to an immutable tuple.
    """

    if isinstance(path, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of states, not a string."
        )

    try:
        frozen = tuple(path)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a finite sequence of states."
        ) from exc

    if not allow_empty and not frozen:
        raise ValueError(f"{name} must be nonempty.")

    for state in frozen:
        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                f"Every state in {name} must be hashable."
            ) from exc

    return frozen


def _freeze_paths(
    paths: Sequence[Sequence[State]] | None,
) -> Paths | None:
    """
    Convert several state paths to an immutable tuple of tuples.
    """

    if paths is None:
        return None

    if isinstance(paths, (str, bytes)):
        raise TypeError(
            "paths must be a sequence of state paths."
        )

    try:
        sequence = tuple(paths)
    except TypeError as exc:
        raise TypeError(
            "paths must be a finite sequence of state paths."
        ) from exc

    return tuple(
        _freeze_path(
            path,
            name=f"paths[{index}]",
            allow_empty=True,
        )
        for index, path in enumerate(sequence)
    )


def _validate_optional_nonnegative_integer(
    value: int | None,
    *,
    name: str,
) -> int | None:
    """
    Validate an optional nonnegative integer.
    """

    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a nonnegative integer or None.")

    result = int(value)

    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")

    return result


def _validate_optional_positive_integer(
    value: int | None,
    *,
    name: str,
) -> int | None:
    """
    Validate an optional positive integer.
    """

    result = _validate_optional_nonnegative_integer(
        value,
        name=name,
    )

    if result is not None and result == 0:
        raise ValueError(f"{name} must be positive.")

    return result


def _validate_optional_finite_float(
    value: float | None,
    *,
    name: str,
    nonnegative: bool = False,
) -> float | None:
    """
    Validate an optional finite real number.
    """

    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real number or None.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a finite real number or None."
        ) from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")

    return result


def _validate_confidence_level(
    value: float | None,
) -> float | None:
    """
    Validate an optional confidence level in the open interval ``(0, 1)``.
    """

    result = _validate_optional_finite_float(
        value,
        name="confidence_level",
    )

    if result is None:
        return None

    if not 0.0 < result < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between zero and one."
        )

    return result


def _validate_confidence_interval(
    value: Sequence[float] | None,
) -> ConfidenceInterval | None:
    """
    Validate an optional two-sided confidence interval.
    """

    if value is None:
        return None

    if isinstance(value, (str, bytes)):
        raise TypeError(
            "confidence_interval must be a two-element real sequence."
        )

    try:
        interval = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "confidence_interval must be a two-element real sequence."
        ) from exc

    if len(interval) != 2:
        raise ValueError(
            "confidence_interval must contain exactly two endpoints."
        )

    lower = _validate_optional_finite_float(
        interval[0],
        name="confidence_interval lower endpoint",
    )
    upper = _validate_optional_finite_float(
        interval[1],
        name="confidence_interval upper endpoint",
    )

    assert lower is not None
    assert upper is not None

    if lower > upper:
        raise ValueError(
            "The lower confidence-interval endpoint cannot exceed "
            "the upper endpoint."
        )

    return lower, upper


def _validate_seed(
    seed: int | None,
) -> int | None:
    """
    Validate an optional NumPy-compatible nonnegative seed.
    """

    return _validate_optional_nonnegative_integer(
        seed,
        name="seed",
    )


def _values_equal(
    left: Any,
    right: Any,
) -> bool:
    """
    Compare result values while supporting NumPy arrays.
    """

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))

    try:
        result = left == right
    except (TypeError, ValueError):
        return False

    if isinstance(result, np.ndarray):
        return bool(np.all(result))

    return bool(result)


# ============================================================================
# Monte Carlo result
# ============================================================================


@dataclass(frozen=True, slots=True, eq=False)
class MonteCarloResult:
    """
    Immutable result of a Monte Carlo computation.

    Parameters
    ----------
    method
        Name of the simulation or estimation method that produced the
        result.

    estimate
        Primary estimate. This may be a scalar, vector, matrix, mapping,
        or another immutable result value.

    samples
        Optional one-dimensional numerical sample array.

    path
        Optional single simulated state path.

    paths
        Optional collection of simulated state paths.

    states
        Optional ordered state labels corresponding to vector or matrix
        estimates.

    variance
        Optional estimated variance. This may be scalar- or array-valued.

    standard_error
        Optional estimated standard error. This may be scalar- or
        array-valued.

    confidence_interval
        Optional scalar confidence interval ``(lower, upper)``.

    confidence_level
        Optional confidence level in ``(0, 1)``.

    n_samples
        Number of samples used in the estimate.

    n_paths
        Number of simulated paths used in the estimate.

    steps
        Number of transition steps per path, where applicable.

    burn_in
        Number of initial observations discarded before estimation.

    thinning
        Observation-thinning interval.

    seed
        Random seed used to initialize the generator, if known.

    rng_name
        Name of the random bit-generator implementation.

    converged
        Optional convergence classification supplied by the producing
        algorithm.

    metadata
        Additional descriptive data.

    Notes
    -----
    The class does not infer statistical quantities. For example, it does
    not derive ``standard_error`` from ``variance`` because later routines
    may use correlated observations, weighted samples, batch means, or
    other estimators.

    Arrays are copied and made read-only. Paths and states are converted to
    tuples, and metadata is stored as a read-only mapping.
    """

    method: str

    estimate: Any = None
    samples: ArrayLike | None = None

    path: Sequence[State] | None = None
    paths: Sequence[Sequence[State]] | None = None
    states: Sequence[State] | None = None

    variance: Any = None
    standard_error: Any = None

    confidence_interval: Sequence[float] | None = None
    confidence_level: float | None = None

    n_samples: int | None = None
    n_paths: int | None = None
    steps: int | None = None
    burn_in: int | None = None
    thinning: int | None = None

    seed: int | None = None
    rng_name: str | None = None
    converged: bool | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and freeze all stored result data.
        """

        if not isinstance(self.method, str):
            raise TypeError("method must be a string.")

        method = self.method.strip()

        if not method:
            raise ValueError("method must be nonempty.")

        object.__setattr__(self, "method", method)

        object.__setattr__(
            self,
            "estimate",
            self._freeze_result_value(
                self.estimate,
                name="estimate",
            ),
        )

        object.__setattr__(
            self,
            "samples",
            _freeze_array(
                self.samples,
                name="samples",
                ndim=1,
                allow_empty=True,
            ),
        )

        frozen_path = (
            None
            if self.path is None
            else _freeze_path(
                self.path,
                name="path",
                allow_empty=True,
            )
        )
        object.__setattr__(self, "path", frozen_path)

        frozen_paths = _freeze_paths(self.paths)
        object.__setattr__(self, "paths", frozen_paths)

        frozen_states = (
            None
            if self.states is None
            else _freeze_path(
                self.states,
                name="states",
                allow_empty=False,
            )
        )

        if frozen_states is not None:
            if len(set(frozen_states)) != len(frozen_states):
                raise ValueError("states must contain unique labels.")

        object.__setattr__(self, "states", frozen_states)

        object.__setattr__(
            self,
            "variance",
            self._freeze_nonnegative_statistic(
                self.variance,
                name="variance",
            ),
        )

        object.__setattr__(
            self,
            "standard_error",
            self._freeze_nonnegative_statistic(
                self.standard_error,
                name="standard_error",
            ),
        )

        interval = _validate_confidence_interval(
            self.confidence_interval
        )
        level = _validate_confidence_level(
            self.confidence_level
        )

        if interval is not None and level is None:
            raise ValueError(
                "confidence_level is required when confidence_interval "
                "is provided."
            )

        if level is not None and interval is None:
            raise ValueError(
                "confidence_interval is required when confidence_level "
                "is provided."
            )

        object.__setattr__(
            self,
            "confidence_interval",
            interval,
        )
        object.__setattr__(
            self,
            "confidence_level",
            level,
        )

        n_samples = _validate_optional_nonnegative_integer(
            self.n_samples,
            name="n_samples",
        )
        n_paths = _validate_optional_nonnegative_integer(
            self.n_paths,
            name="n_paths",
        )
        steps = _validate_optional_nonnegative_integer(
            self.steps,
            name="steps",
        )
        burn_in = _validate_optional_nonnegative_integer(
            self.burn_in,
            name="burn_in",
        )
        thinning = _validate_optional_positive_integer(
            self.thinning,
            name="thinning",
        )

        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "n_paths", n_paths)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "burn_in", burn_in)
        object.__setattr__(self, "thinning", thinning)

        object.__setattr__(
            self,
            "seed",
            _validate_seed(self.seed),
        )

        if self.rng_name is not None:
            if not isinstance(self.rng_name, str):
                raise TypeError("rng_name must be a string or None.")

            rng_name = self.rng_name.strip()

            if not rng_name:
                raise ValueError(
                    "rng_name must be nonempty when provided."
                )

            object.__setattr__(self, "rng_name", rng_name)

        if self.converged is not None and not isinstance(
            self.converged,
            (bool, np.bool_),
        ):
            raise TypeError("converged must be a boolean or None.")

        if isinstance(self.converged, np.bool_):
            object.__setattr__(
                self,
                "converged",
                bool(self.converged),
            )

        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata),
        )

        self._validate_internal_consistency()

    # ------------------------------------------------------------------
    # Result-value freezing
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_result_value(
        value: Any,
        *,
        name: str,
    ) -> Any:
        """
        Freeze common mutable result values.

        NumPy arrays are copied and made read-only. Mappings are copied
        into read-only mapping proxies. Lists are converted recursively
        to tuples. Scalar values are retained.
        """

        if value is None:
            return None

        if isinstance(value, np.ndarray):
            return _freeze_array(
                value,
                name=name,
            )

        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    key: MonteCarloResult._freeze_result_value(
                        item,
                        name=f"{name}[{key!r}]",
                    )
                    for key, item in value.items()
                }
            )

        if isinstance(value, list):
            return tuple(
                MonteCarloResult._freeze_result_value(
                    item,
                    name=f"{name}[{index}]",
                )
                for index, item in enumerate(value)
            )

        if isinstance(value, tuple):
            return tuple(
                MonteCarloResult._freeze_result_value(
                    item,
                    name=f"{name}[{index}]",
                )
                for index, item in enumerate(value)
            )

        return value

    @classmethod
    def _freeze_nonnegative_statistic(
        cls,
        value: Any,
        *,
        name: str,
    ) -> Any:
        """
        Freeze and validate a nonnegative scalar or array statistic.
        """

        if value is None:
            return None

        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"{name} must be a nonnegative real value, "
                "array, or None."
            )

        if np.isscalar(value):
            result = _validate_optional_finite_float(
                value,
                name=name,
                nonnegative=True,
            )
            return result

        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} must be a nonnegative real value, "
                "array, or None."
            ) from exc

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must contain only finite values."
            )

        if np.any(array < 0.0):
            raise ValueError(
                f"{name} must contain only nonnegative values."
            )

        return _freeze_array(
            array,
            name=name,
        )

    # ------------------------------------------------------------------
    # Consistency validation
    # ------------------------------------------------------------------

    def _validate_internal_consistency(self) -> None:
        """
        Validate relationships among optional fields.
        """

        if self.path is not None and self.paths is not None:
            raise ValueError(
                "Specify either path or paths, not both."
            )

        if self.path is not None:
            inferred_steps = max(len(self.path) - 1, 0)

            if self.steps is not None and self.steps != inferred_steps:
                raise ValueError(
                    "steps must equal len(path) - 1 when a path is "
                    "provided."
                )

            if self.n_paths is not None and self.n_paths != 1:
                raise ValueError(
                    "n_paths must equal one when path is provided."
                )

        if self.paths is not None:
            inferred_n_paths = len(self.paths)

            if (
                self.n_paths is not None
                and self.n_paths != inferred_n_paths
            ):
                raise ValueError(
                    "n_paths must equal len(paths) when paths are "
                    "provided."
                )

            lengths = {len(path) for path in self.paths}

            if self.steps is not None and lengths:
                expected_length = self.steps + 1

                if any(
                    length != expected_length
                    for length in lengths
                ):
                    raise ValueError(
                        "Every path must contain steps + 1 states."
                    )

        if self.samples is not None:
            inferred_n_samples = int(self.samples.size)

            if (
                self.n_samples is not None
                and self.n_samples != inferred_n_samples
            ):
                raise ValueError(
                    "n_samples must equal samples.size when samples "
                    "are provided."
                )

        if (
            self.burn_in is not None
            and self.steps is not None
            and self.burn_in > self.steps
        ):
            raise ValueError(
                "burn_in cannot exceed the number of transition steps."
            )

    # ------------------------------------------------------------------
    # Presence properties
    # ------------------------------------------------------------------

    @property
    def has_estimate(self) -> bool:
        """
        Whether a primary estimate is stored.
        """

        return self.estimate is not None

    @property
    def has_samples(self) -> bool:
        """
        Whether numerical samples are stored.
        """

        return self.samples is not None

    @property
    def has_path(self) -> bool:
        """
        Whether a single simulated path is stored.
        """

        return self.path is not None

    @property
    def has_paths(self) -> bool:
        """
        Whether several simulated paths are stored.
        """

        return self.paths is not None

    @property
    def has_uncertainty(self) -> bool:
        """
        Whether any uncertainty statistic is stored.
        """

        return any(
            value is not None
            for value in (
                self.variance,
                self.standard_error,
                self.confidence_interval,
            )
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def interval_width(self) -> float | None:
        """
        Width of the stored confidence interval.
        """

        if self.confidence_interval is None:
            return None

        lower, upper = self.confidence_interval
        return upper - lower

    @property
    def effective_sample_size(self) -> int | None:
        """
        Number of retained samples after burn-in and thinning.

        Notes
        -----
        This is a bookkeeping count, not the statistical effective sample
        size that corrects for serial autocorrelation.
        """

        if self.n_samples is not None:
            return self.n_samples

        if self.steps is None:
            return None

        burn_in = self.burn_in or 0
        thinning = self.thinning or 1

        retained_observations = self.steps + 1 - burn_in

        if retained_observations <= 0:
            return 0

        return (
            retained_observations + thinning - 1
        ) // thinning

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def with_metadata(
        self,
        **updates: Any,
    ) -> MonteCarloResult:
        """
        Return a copy with updated metadata.
        """

        metadata = dict(self.metadata)
        metadata.update(updates)

        return replace(
            self,
            metadata=metadata,
        )

    def renamed(
        self,
        method: str,
    ) -> MonteCarloResult:
        """
        Return a copy with a different method name.
        """

        return replace(
            self,
            method=method,
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Return a compact mutable summary dictionary.
        """

        return {
            "method": self.method,
            "has_estimate": self.has_estimate,
            "has_samples": self.has_samples,
            "has_path": self.has_path,
            "has_paths": self.has_paths,
            "has_uncertainty": self.has_uncertainty,
            "n_samples": self.n_samples,
            "n_paths": self.n_paths,
            "steps": self.steps,
            "burn_in": self.burn_in,
            "thinning": self.thinning,
            "effective_sample_size": self.effective_sample_size,
            "seed": self.seed,
            "rng_name": self.rng_name,
            "converged": self.converged,
            "confidence_level": self.confidence_level,
            "confidence_interval": self.confidence_interval,
            "interval_width": self.interval_width,
            "metadata": dict(self.metadata),
        }

    # ------------------------------------------------------------------
    # Equality and representation
    # ------------------------------------------------------------------

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(other, MonteCarloResult):
            return NotImplemented

        scalar_fields = (
            "method",
            "path",
            "paths",
            "states",
            "confidence_interval",
            "confidence_level",
            "n_samples",
            "n_paths",
            "steps",
            "burn_in",
            "thinning",
            "seed",
            "rng_name",
            "converged",
        )

        if any(
            getattr(self, name) != getattr(other, name)
            for name in scalar_fields
        ):
            return False

        if not _values_equal(self.estimate, other.estimate):
            return False

        if not _values_equal(self.samples, other.samples):
            return False

        if not _values_equal(self.variance, other.variance):
            return False

        if not _values_equal(
            self.standard_error,
            other.standard_error,
        ):
            return False

        return dict(self.metadata) == dict(other.metadata)

    def __repr__(self) -> str:
        attributes = [
            f"method={self.method!r}",
        ]

        if self.n_samples is not None:
            attributes.append(f"n_samples={self.n_samples}")

        if self.n_paths is not None:
            attributes.append(f"n_paths={self.n_paths}")

        if self.steps is not None:
            attributes.append(f"steps={self.steps}")

        if self.seed is not None:
            attributes.append(f"seed={self.seed}")

        if self.has_estimate:
            attributes.append("estimate=<stored>")

        if self.has_uncertainty:
            attributes.append("uncertainty=<stored>")

        return (
            f"{type(self).__name__}("
            + ", ".join(attributes)
            + ")"
        )


__all__ = [
    "ConfidenceInterval",
    "MonteCarloResult",
    "Path",
    "Paths",
    "State",
]
