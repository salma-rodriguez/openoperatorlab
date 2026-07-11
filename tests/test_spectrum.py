import numpy as np

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.spectrum import (
    SpectralAnalyzer,
    SpectralPartition,
    ResolventAnalyzer,
    SpectralStatistics,
)


def test_spectral_analyzer_delegates_eigenvalues():
    A = LinearOperator(np.diag([3, 1, 2]))
    S = SpectralAnalyzer(A)

    assert np.allclose(np.sort(S.eigenvalues()), np.array([1, 2, 3]))


def test_sorted_eigenvalues_abs():
    A = LinearOperator(np.diag([-3, 1, -2]))
    S = SpectralAnalyzer(A)

    assert np.allclose(S.sorted_eigenvalues("abs"), np.array([1, -2, -3]))


def test_spectral_partition_sizes():
    vals = np.arange(10)
    P = SpectralPartition(vals, alpha=0.2)

    assert P.sizes() == (2, 6, 2)


def test_spectral_analyzer_partition():
    A = LinearOperator(np.diag(np.arange(10)))
    S = SpectralAnalyzer(A)
    P = S.partition(alpha=0.2)

    assert P.sizes() == (2, 6, 2)


def test_resolvent_matrix():
    A = LinearOperator(np.diag([1.0, 2.0]))
    R = ResolventAnalyzer(A)

    expected = np.diag([1 / (1 - 0.5), 1 / (2 - 0.5)])

    assert np.allclose(R.matrix(0.5), expected)


def test_resolvent_trace():
    A = LinearOperator(np.diag([1.0, 2.0]))
    R = ResolventAnalyzer(A)

    expected = 1 / (1 - 0.5) + 1 / (2 - 0.5)

    assert np.isclose(R.trace(0.5), expected)


def test_resolvent_determinant():
    A = LinearOperator(np.diag([1.0, 2.0]))
    R = ResolventAnalyzer(A)

    expected = (1 - 0.5) * (2 - 0.5)

    assert np.isclose(R.determinant(0.5), expected)


def test_spectral_statistics_spacings():
    vals = np.array([1, 4, 9])
    stats = SpectralStatistics(vals)

    assert np.allclose(stats.spacings(), np.array([3, 5]))


def test_spectral_statistics_normalized_spacings():
    vals = np.array([1, 4, 9])
    stats = SpectralStatistics(vals)

    assert np.allclose(stats.normalized_spacings(), np.array([0.75, 1.25]))


def test_spectral_statistics_summary():
    vals = np.array([1, 4, 9])
    stats = SpectralStatistics(vals)
    summary = stats.summary()

    assert summary["count"] == 3
    assert summary["num_spacings"] == 2
    assert np.isclose(summary["mean_spacing"], 4.0)
