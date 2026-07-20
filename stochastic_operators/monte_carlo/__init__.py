"""
Monte Carlo simulation and empirical estimation.

The subpackage is organized into four layers:

- result: immutable result objects;
- simulation: trajectory generation;
- empirical: empirical estimators;
- statistics: statistical summaries and confidence intervals.
"""

from .result import (
    ConfidenceInterval,
    MonteCarloResult,
    Path,
    Paths,
    State,
)
from .simulation import (
    simulate_chain,
    simulate_paths,
)


__all__ = [
    "ConfidenceInterval",
    "MonteCarloResult",
    "Path",
    "Paths",
    "State",
    "simulate_chain",
    "simulate_paths",
]
