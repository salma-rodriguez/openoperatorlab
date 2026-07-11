import numpy as np

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.diagnostics import (
    OperatorDiagnostics,
    ComparativeDiagnostics,
    StabilityDiagnostics,
    DiagnosticReport,
)


def test_operator_diagnostics_algebra_summary():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    D = OperatorDiagnostics(A)

    summary = D.algebra_summary()

    assert summary["name"] == "A"
    assert summary["shape"] == (2, 2)
    assert summary["rank"] == 2
    assert np.isclose(summary["trace"], 3.0)
    assert np.isclose(summary["det"], 2.0)


def test_operator_diagnostics_spectral_summary():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    D = OperatorDiagnostics(A)

    summary = D.spectral_summary()

    assert summary["num_eigenvalues"] == 2
    assert "spacing_summary" in summary


def test_operator_diagnostics_geometry_summary():
    A = LinearOperator(np.eye(2), name="I")
    D = OperatorDiagnostics(A)

    summary = D.geometry_summary()

    assert summary["operator"] == "I"
    assert "symmetry" in summary
    assert "boundary" in summary
    assert "locality" in summary


def test_operator_diagnostics_summary():
    A = LinearOperator(np.eye(2), name="I")
    D = OperatorDiagnostics(A)

    summary = D.summary()

    assert "algebra" in summary
    assert "spectral" in summary
    assert "geometry" in summary
    assert "hilbert_polya" in summary


def test_comparative_diagnostics_norm_table():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    C = ComparativeDiagnostics([A, B])
    table = C.norm_table("fro")

    assert np.isclose(table["A"], np.sqrt(2))
    assert np.isclose(table["B"], 2 * np.sqrt(2))


def test_comparative_diagnostics_pairwise_distances():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    C = ComparativeDiagnostics([A, B])
    distances = C.pairwise_distances("fro")

    assert np.isclose(distances[("A", "B")], np.sqrt(2))


def test_comparative_diagnostics_summary():
    A = LinearOperator(np.eye(2), name="A")
    B = LinearOperator(2 * np.eye(2), name="B")

    C = ComparativeDiagnostics([A, B])
    summary = C.summary()

    assert summary["num_operators"] == 2
    assert summary["names"] == ("A", "B")
    assert "frobenius_norms" in summary
    assert "pairwise_frobenius_distances" in summary


def test_stability_diagnostics_spectral_radius():
    A = LinearOperator(np.diag([1.0, -3.0, 2.0]), name="A")
    S = StabilityDiagnostics(A)

    assert np.isclose(S.spectral_radius(), 3.0)


def test_stability_diagnostics_real_spectrum_defect():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    S = StabilityDiagnostics(A)

    assert np.isclose(S.real_spectrum_defect(), 0.0)


def test_stability_diagnostics_normality_defect():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    S = StabilityDiagnostics(A)

    assert np.isclose(S.normality_defect(), 0.0)


def test_stability_diagnostics_summary():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    S = StabilityDiagnostics(A)

    summary = S.summary()

    assert "condition_number" in summary
    assert "spectral_radius" in summary
    assert "real_spectrum_defect" in summary
    assert "normality_defect" in summary


def test_diagnostic_report_without_zeta():
    A = LinearOperator(np.eye(2), name="I")
    R = DiagnosticReport(A, include_zeta=False)

    report = R.generate()

    assert report["operator"] == "I"
    assert "diagnostics" in report
    assert "stability" in report
    assert "spectral_zeta" not in report


def test_diagnostic_report_with_zeta():
    A = LinearOperator(np.eye(2), name="I")
    R = DiagnosticReport(A, include_zeta=True)

    report = R.generate()

    assert report["operator"] == "I"
    assert "spectral_zeta" in report
