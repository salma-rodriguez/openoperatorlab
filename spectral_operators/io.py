"""
spectral_operator.io
====================

Input/output utilities for spectral operator objects.

This module provides lightweight serialization and deserialization helpers
for LinearOperator objects, diagnostic reports, and JSON-compatible data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core.algebra import LinearOperator, OperatorError


# ===========================================================================
# JSON Utilities
# ===========================================================================

class JSONSerializer:
    """
    JSON serialization utilities.
    """

    @staticmethod
    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, complex):
            return {
                "__complex__": True,
                "real": obj.real,
                "imag": obj.imag,
            }

        if isinstance(obj, tuple):
            return list(obj)

        if isinstance(obj, dict):
            return {
                str(k): JSONSerializer.to_jsonable(v)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                JSONSerializer.to_jsonable(v)
                for v in obj
            ]

        return obj

    @staticmethod
    def dumps(obj, **kwargs) -> str:
        return json.dumps(
            JSONSerializer.to_jsonable(obj),
            **kwargs,
        )

    @staticmethod
    def dump(obj, path) -> None:
        path = Path(path)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                JSONSerializer.to_jsonable(obj),
                f,
                indent=2,
            )

    @staticmethod
    def load(path):
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


# ===========================================================================
# Operator IO
# ===========================================================================

class OperatorIO:
    """
    Save and load LinearOperator objects.
    """

    @staticmethod
    def to_dict(operator: LinearOperator) -> dict:
        if not isinstance(operator, LinearOperator):
            raise OperatorError("operator must be a LinearOperator.")

        return {
            "name": operator.name,
            "matrix": JSONSerializer.to_jsonable(operator.matrix),
            "metadata": JSONSerializer.to_jsonable(operator.metadata),
        }

    @staticmethod
    def from_dict(data: dict) -> LinearOperator:
        if not isinstance(data, dict):
            raise OperatorError("data must be a dictionary.")

        matrix = np.asarray(data["matrix"])

        return LinearOperator(
            matrix=matrix,
            name=data.get("name", "LinearOperator"),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def save_json(operator: LinearOperator, path) -> None:
        JSONSerializer.dump(
            OperatorIO.to_dict(operator),
            path,
        )

    @staticmethod
    def load_json(path) -> LinearOperator:
        return OperatorIO.from_dict(
            JSONSerializer.load(path)
        )


# ===========================================================================
# Diagnostic IO
# ===========================================================================

class DiagnosticIO:
    """
    Save diagnostic dictionaries and reports.
    """

    @staticmethod
    def save(report: dict, path) -> None:
        if not isinstance(report, dict):
            raise OperatorError("report must be a dictionary.")

        JSONSerializer.dump(report, path)

    @staticmethod
    def load(path) -> dict:
        return JSONSerializer.load(path)


# ===========================================================================
# Numpy IO
# ===========================================================================

class NumpyIO:
    """
    Save and load NumPy array data.
    """

    @staticmethod
    def save_array(array, path) -> None:
        path = Path(path)
        np.save(path, np.asarray(array))

    @staticmethod
    def load_array(path) -> np.ndarray:
        path = Path(path)
        return np.load(path)

    @staticmethod
    def save_npz(path, **arrays) -> None:
        path = Path(path)
        np.savez(
            path,
            **{
                key: np.asarray(value)
                for key, value in arrays.items()
            },
        )

    @staticmethod
    def load_npz(path) -> dict:
        path = Path(path)

        with np.load(path) as data:
            return {
                key: data[key]
                for key in data.files
            }
