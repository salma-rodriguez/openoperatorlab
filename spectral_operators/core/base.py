# spectral_operators/core/base.py

"""
spectral_operators.core.base
============================

Abstract foundations for OperatorLab operator objects.
"""

from __future__ import annotations

from abc import ABC


class OperatorBase(ABC):
    """
    Base class for operator-like mathematical objects.

    The interface will be expanded as additional operator families
    are introduced.
    """

    __slots__ = ()
