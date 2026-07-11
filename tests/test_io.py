import json
import numpy as np

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.io import (
    JSONSerializer,
    OperatorIO,
    DiagnosticIO,
    NumpyIO,
)


def test_json_serializer_numpy_array():
    arr = np.array([1, 2, 3])
    assert JSONSerializer.to_jsonable(arr) == [1, 2, 3]


def test_json_serializer_numpy_scalar():
    x = np.float64(3.5)
    assert JSONSerializer.to_jsonable(x) == 3.5


def test_json_serializer_complex():
    z = 1 + 2j
    result = JSONSerializer.to_jsonable(z)

    assert result == {
        "__complex__": True,
        "real": 1.0,
        "imag": 2.0,
    }


def test_json_serializer_dumps():
    obj = {"x": np.array([1, 2])}
    s = JSONSerializer.dumps(obj)

    assert json.loads(s) == {"x": [1, 2]}


def test_json_serializer_dump_and_load(tmp_path):
    path = tmp_path / "data.json"
    obj = {"x": np.array([1, 2, 3])}

    JSONSerializer.dump(obj, path)
    loaded = JSONSerializer.load(path)

    assert loaded == {"x": [1, 2, 3]}


def test_operator_io_to_dict():
    A = LinearOperator(np.eye(2), name="I", metadata={"kind": "identity"})
    data = OperatorIO.to_dict(A)

    assert data["name"] == "I"
    assert data["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert data["metadata"] == {"kind": "identity"}


def test_operator_io_from_dict():
    data = {
        "name": "A",
        "matrix": [[1, 2], [3, 4]],
        "metadata": {"source": "test"},
    }

    A = OperatorIO.from_dict(data)

    assert A.name == "A"
    assert np.allclose(A.matrix, np.array([[1, 2], [3, 4]]))
    assert A.metadata["source"] == "test"


def test_operator_io_save_and_load_json(tmp_path):
    path = tmp_path / "operator.json"
    A = LinearOperator(np.eye(2), name="I")

    OperatorIO.save_json(A, path)
    loaded = OperatorIO.load_json(path)

    assert loaded.name == "I"
    assert np.allclose(loaded.matrix, np.eye(2))


def test_diagnostic_io_save_and_load(tmp_path):
    path = tmp_path / "diagnostic.json"
    report = {"operator": "A", "rank": 2}

    DiagnosticIO.save(report, path)
    loaded = DiagnosticIO.load(path)

    assert loaded == report


def test_numpy_io_save_and_load_array(tmp_path):
    path = tmp_path / "array.npy"
    arr = np.array([[1, 2], [3, 4]])

    NumpyIO.save_array(arr, path)
    loaded = NumpyIO.load_array(path)

    assert np.allclose(loaded, arr)


def test_numpy_io_save_and_load_npz(tmp_path):
    path = tmp_path / "arrays.npz"
    a = np.array([1, 2])
    b = np.array([3, 4])

    NumpyIO.save_npz(path, a=a, b=b)
    loaded = NumpyIO.load_npz(path)

    assert np.allclose(loaded["a"], a)
    assert np.allclose(loaded["b"], b)
