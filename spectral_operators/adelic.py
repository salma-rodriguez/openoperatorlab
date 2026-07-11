"""
spectral_operator.adelic
========================

Adelic-style constructions for spectral operator models.

This module contains tools for assembling local components into
global operators using prime-indexed, place-indexed, or otherwise
labeled systems.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator, OperatorError
from .operators import AdelicOperator
from .weights import AdelicWeight, PrimeWeight


# ===========================================================================
# Local Components
# ===========================================================================

class LocalComponent:
    """
    Labeled local operator component.

    Parameters
    ----------
    label
        Label identifying the local component, such as a prime,
        place, scale, or region.

    operator : LinearOperator
        Local operator associated with the label.
    """

    def __init__(self, label, operator: LinearOperator):

        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        self.label = label
        self.operator = operator

    @property
    def shape(self):
        return self.operator.shape

    @property
    def matrix(self):
        return self.operator.matrix

    def as_tuple(self):
        return self.label, self.operator


# ===========================================================================
# Adelic System
# ===========================================================================

class AdelicSystem:
    """
    Collection of local components and associated weights.
    """

    def __init__(
        self,
        components,
        *,
        weights=None,
        normalize: bool = True,
    ):

        if len(components) == 0:
            raise OperatorError("AdelicSystem requires at least one component.")

        for component in components:
            if not isinstance(component, LocalComponent):
                raise OperatorError(
                    "components must be LocalComponent instances."
                )

        shape = components[0].shape

        for component in components:
            if component.shape != shape:
                raise OperatorError(
                    "all local components must have the same shape."
                )

        labels = tuple(component.label for component in components)

        if weights is None:
            weight_system = AdelicWeight(
                labels=labels,
                normalize=normalize,
            )

        elif isinstance(weights, AdelicWeight):
            weight_system = weights

        else:
            weight_system = AdelicWeight(
                labels=labels,
                weights=weights,
                normalize=normalize,
            )

        if tuple(weight_system.labels) != labels:
            raise OperatorError(
                "weight labels must match component labels."
            )

        self.components = tuple(components)
        self.labels = labels
        self.weights = weight_system
        self.shape = shape

    @classmethod
    def from_operators(
        cls,
        operators,
        *,
        labels=None,
        weights=None,
        normalize: bool = True,
    ) -> "AdelicSystem":

        if labels is None:
            labels = tuple(range(len(operators)))

        if len(labels) != len(operators):
            raise OperatorError("labels must match operators.")

        components = [
            LocalComponent(label, operator)
            for label, operator in zip(labels, operators)
        ]

        return cls(
            components,
            weights=weights,
            normalize=normalize,
        )

    def local_operators(self) -> list[LinearOperator]:
        return [component.operator for component in self.components]

    def weight_array(self) -> np.ndarray:
        return self.weights.as_array()


# ===========================================================================
# Adelic Builder
# ===========================================================================

class AdelicBuilder:
    """
    Builder for assembling adelic-style global operators.
    """

    def __init__(self, system: AdelicSystem):

        if not isinstance(system, AdelicSystem):
            raise OperatorError("system must be an AdelicSystem.")

        self.system = system

    def build(self, *, name: str | None = None) -> AdelicOperator:
        """
        Build a global AdelicOperator from the local system.
        """

        if name is None:
            name = "AdelicOperator"

        return AdelicOperator(
            self.system.local_operators(),
            weights=self.system.weight_array(),
            labels=self.system.labels,
            normalize=False,
            name=name,
        )


# ===========================================================================
# Adelic Analyzer
# ===========================================================================

class AdelicAnalyzer:
    """
    Diagnostics for adelic-style operator systems.
    """

    def __init__(self, system: AdelicSystem):

        if not isinstance(system, AdelicSystem):
            raise OperatorError("system must be an AdelicSystem.")

        self.system = system

    def component_norms(self, kind: str = "fro") -> dict:
        """
        Return norms of local components.
        """

        return {
            component.label: component.operator.norm(kind)
            for component in self.system.components
        }

    def weighted_component_norms(self, kind: str = "fro") -> dict:
        """
        Return weighted local component norms.
        """

        weights = self.system.weight_array()

        return {
            component.label: float(abs(w) * component.operator.norm(kind))
            for component, w in zip(self.system.components, weights)
        }

    def weight_summary(self) -> dict:
        """
        Return information about the adelic weights.
        """

        weights = self.system.weight_array()

        return {
            "labels": self.system.labels,
            "weights": tuple(weights.tolist()),
            "sum_abs_weights": float(np.sum(np.abs(weights))),
            "num_components": len(weights),
        }

    def summary(self) -> dict:
        """
        Return summary diagnostics for the adelic system.
        """

        return {
            "shape": self.system.shape,
            "labels": self.system.labels,
            "component_norms": self.component_norms(),
            "weighted_component_norms": self.weighted_component_norms(),
            "weights": self.weight_summary(),
        }
