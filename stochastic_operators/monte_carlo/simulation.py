"""
Primitive Monte Carlo simulation.

This module provides trajectory generation for finite-state stochastic
kernels. It deliberately contains no empirical estimators or statistical
summaries.

All higher-level Monte Carlo functionality should build upon the simulation
primitives defined here.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .result import MonteCarloResult


State: TypeAlias = Hashable
ProbabilityArray: TypeAlias = NDArray[np.float64]


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a nonnegative integer.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a nonnegative integer.")

    result = int(value)

    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")

    return result


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a positive integer.
    """

    result = _validate_nonnegative_integer(
        value,
        name=name,
    )

    if result == 0:
        raise ValueError(f"{name} must be positive.")

    return result


def _validate_rng(
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.random.Generator, int | None]:
    """
    Resolve a local NumPy random generator.

    Parameters
    ----------
    seed
        Optional nonnegative integer seed.

    rng
        Optional existing NumPy random generator.

    Returns
    -------
    tuple
        ``(generator, recorded_seed)``.

    Notes
    -----
    An explicitly supplied generator takes precedence over ``seed``. To
    avoid ambiguous reproducibility metadata, supplying both is rejected.
    """

    if rng is not None and seed is not None:
        raise ValueError("Specify either seed or rng, not both.")

    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be an instance of numpy.random.Generator."
            )

        return rng, None

    if seed is not None:
        seed = _validate_nonnegative_integer(
            seed,
            name="seed",
        )

    return np.random.default_rng(seed), seed


def _validate_states(
    states: Sequence[State],
) -> tuple[State, ...]:
    """
    Validate and freeze an ordered finite state space.
    """

    if isinstance(states, (str, bytes)):
        raise TypeError(
            "states must be a sequence of state labels, not a string."
        )

    try:
        frozen = tuple(states)
    except TypeError as exc:
        raise TypeError(
            "states must be a finite sequence of state labels."
        ) from exc

    if not frozen:
        raise ValueError("states must be nonempty.")

    for state in frozen:
        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                "Every state label must be hashable."
            ) from exc

    if len(set(frozen)) != len(frozen):
        raise ValueError("states must contain unique labels.")

    return frozen


def _validate_transition_matrix(
    matrix: ArrayLike,
    *,
    n_states: int,
    atol: float = 1e-12,
) -> ProbabilityArray:
    """
    Validate a finite row-stochastic transition matrix.
    """

    try:
        transition_matrix = np.asarray(
            matrix,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "transition_matrix must be convertible to a real NumPy array."
        ) from exc

    expected_shape = (n_states, n_states)

    if transition_matrix.shape != expected_shape:
        raise ValueError(
            "transition_matrix must have shape "
            f"{expected_shape}, but received "
            f"{transition_matrix.shape}."
        )

    if not np.all(np.isfinite(transition_matrix)):
        raise ValueError(
            "transition_matrix must contain only finite values."
        )

    if np.any(transition_matrix < -atol):
        raise ValueError(
            "transition_matrix cannot contain negative probabilities."
        )

    # Remove harmless negative roundoff before normalization checks.
    transition_matrix = np.where(
        transition_matrix < 0.0,
        0.0,
        transition_matrix,
    )

    row_sums = transition_matrix.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "Every transition-matrix row must sum to one."
        )

    frozen = np.array(
        transition_matrix,
        dtype=np.float64,
        copy=True,
    )
    frozen.setflags(write=False)

    return frozen


def _validate_initial_state(
    initial_state: State,
    *,
    state_to_index: Mapping[State, int],
) -> State:
    """
    Validate that an initial state belongs to the state space.
    """

    try:
        belongs = initial_state in state_to_index
    except TypeError as exc:
        raise TypeError(
            "initial_state must be hashable."
        ) from exc

    if not belongs:
        raise ValueError(
            f"Unknown initial_state: {initial_state!r}."
        )

    return initial_state


def _validate_initial_distribution(
    distribution: ArrayLike,
    *,
    n_states: int,
    atol: float = 1e-12,
) -> ProbabilityArray:
    """
    Validate an initial probability distribution.
    """

    try:
        probabilities = np.asarray(
            distribution,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "initial_distribution must be convertible to a real array."
        ) from exc

    if probabilities.shape != (n_states,):
        raise ValueError(
            "initial_distribution must have shape "
            f"({n_states},), but received {probabilities.shape}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "initial_distribution must contain only finite values."
        )

    if np.any(probabilities < -atol):
        raise ValueError(
            "initial_distribution cannot contain negative probabilities."
        )

    probabilities = np.where(
        probabilities < 0.0,
        0.0,
        probabilities,
    )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "initial_distribution must sum to one."
        )

    frozen = np.array(
        probabilities,
        dtype=np.float64,
        copy=True,
    )
    frozen.setflags(write=False)

    return frozen


def _resolve_initial_state(
    *,
    states: tuple[State, ...],
    state_to_index: Mapping[State, int],
    initial_state: State | None,
    initial_distribution: ArrayLike | None,
    rng: np.random.Generator,
) -> State:
    """
    Resolve a fixed or randomly sampled initial state.
    """

    if initial_state is not None and initial_distribution is not None:
        raise ValueError(
            "Specify either initial_state or initial_distribution, "
            "not both."
        )

    if initial_state is None and initial_distribution is None:
        raise ValueError(
            "Either initial_state or initial_distribution is required."
        )

    if initial_state is not None:
        return _validate_initial_state(
            initial_state,
            state_to_index=state_to_index,
        )

    probabilities = _validate_initial_distribution(
        initial_distribution,
        n_states=len(states),
    )

    index = int(
        rng.choice(
            len(states),
            p=probabilities,
        )
    )

    return states[index]


# ============================================================================
# Sampling primitive
# ============================================================================


def _sample_next_state(
    current_state: State,
    *,
    states: tuple[State, ...],
    state_to_index: Mapping[State, int],
    transition_matrix: ProbabilityArray,
    rng: np.random.Generator,
) -> State:
    """
    Sample one transition from the current state.
    """

    current_index = state_to_index[current_state]
    probabilities = transition_matrix[current_index]

    next_index = int(
        rng.choice(
            len(states),
            p=probabilities,
        )
    )

    return states[next_index]


# ============================================================================
# Public simulation functions
# ============================================================================


def simulate_chain(
    transition_matrix: ArrayLike,
    states: Sequence[State],
    *,
    steps: int,
    initial_state: State | None = None,
    initial_distribution: ArrayLike | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MonteCarloResult:
    """
    Simulate one finite-state, discrete-time Markov chain.

    Parameters
    ----------
    transition_matrix
        Square row-stochastic transition matrix.

    states
        Ordered labels corresponding to the rows and columns of the
        transition matrix.

    steps
        Number of transitions to simulate.

    initial_state
        Fixed initial state.

    initial_distribution
        Probability distribution from which the initial state is sampled.

    seed
        Optional seed used to create a local random generator.

    rng
        Optional existing NumPy random generator.

    metadata
        Optional descriptive metadata.

    Returns
    -------
    MonteCarloResult
        Immutable result containing a path of length ``steps + 1``.

    Notes
    -----
    Exactly one of ``initial_state`` and ``initial_distribution`` must be
    supplied.
    """

    steps = _validate_nonnegative_integer(
        steps,
        name="steps",
    )

    frozen_states = _validate_states(states)

    matrix = _validate_transition_matrix(
        transition_matrix,
        n_states=len(frozen_states),
    )

    generator, recorded_seed = _validate_rng(
        seed=seed,
        rng=rng,
    )

    state_to_index = {
        state: index
        for index, state in enumerate(frozen_states)
    }

    current_state = _resolve_initial_state(
        states=frozen_states,
        state_to_index=state_to_index,
        initial_state=initial_state,
        initial_distribution=initial_distribution,
        rng=generator,
    )

    path: list[State] = [current_state]

    for _ in range(steps):
        current_state = _sample_next_state(
            current_state,
            states=frozen_states,
            state_to_index=state_to_index,
            transition_matrix=matrix,
            rng=generator,
        )
        path.append(current_state)

    result_metadata = {
        "simulation": "discrete_time_markov_chain",
        "state_count": len(frozen_states),
    }

    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")

        result_metadata.update(metadata)

    return MonteCarloResult(
        method="simulate_chain",
        path=tuple(path),
        states=frozen_states,
        steps=steps,
        n_paths=1,
        seed=recorded_seed,
        rng_name=type(generator.bit_generator).__name__,
        metadata=result_metadata,
    )


def simulate_paths(
    transition_matrix: ArrayLike,
    states: Sequence[State],
    *,
    steps: int,
    n_paths: int,
    initial_state: State | None = None,
    initial_distribution: ArrayLike | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MonteCarloResult:
    """
    Simulate several independent finite-state Markov-chain paths.

    Parameters
    ----------
    transition_matrix
        Square row-stochastic transition matrix.

    states
        Ordered state labels.

    steps
        Number of transitions in every path.

    n_paths
        Number of paths to generate.

    initial_state
        Fixed initial state for every path.

    initial_distribution
        Distribution independently sampled at the beginning of every path.

    seed
        Optional seed used to create a local random generator.

    rng
        Optional existing NumPy random generator.

    metadata
        Optional descriptive metadata.

    Returns
    -------
    MonteCarloResult
        Immutable result containing ``n_paths`` paths.
    """

    steps = _validate_nonnegative_integer(
        steps,
        name="steps",
    )
    n_paths = _validate_positive_integer(
        n_paths,
        name="n_paths",
    )

    frozen_states = _validate_states(states)

    matrix = _validate_transition_matrix(
        transition_matrix,
        n_states=len(frozen_states),
    )

    generator, recorded_seed = _validate_rng(
        seed=seed,
        rng=rng,
    )

    state_to_index = {
        state: index
        for index, state in enumerate(frozen_states)
    }

    paths: list[tuple[State, ...]] = []

    for _ in range(n_paths):
        current_state = _resolve_initial_state(
            states=frozen_states,
            state_to_index=state_to_index,
            initial_state=initial_state,
            initial_distribution=initial_distribution,
            rng=generator,
        )

        path: list[State] = [current_state]

        for _ in range(steps):
            current_state = _sample_next_state(
                current_state,
                states=frozen_states,
                state_to_index=state_to_index,
                transition_matrix=matrix,
                rng=generator,
            )
            path.append(current_state)

        paths.append(tuple(path))

    result_metadata = {
        "simulation": "discrete_time_markov_chain",
        "state_count": len(frozen_states),
        "independent_paths": True,
    }

    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")

        result_metadata.update(metadata)

    return MonteCarloResult(
        method="simulate_paths",
        paths=tuple(paths),
        states=frozen_states,
        steps=steps,
        n_paths=n_paths,
        seed=recorded_seed,
        rng_name=type(generator.bit_generator).__name__,
        metadata=result_metadata,
    )


__all__ = [
    "simulate_chain",
    "simulate_paths",
]
