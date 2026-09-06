import numpy as np
import pytest
from ffrefine.fit.linear import asymmetry_metric, fit_cosine_series, is_asymmetric
from ffrefine.mm.analytic import series_energy


def test_symmetric_profile_has_near_zero_metric():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    assert asymmetry_metric(phi, target) < 1e-8
    assert not is_asymmetric(phi, target)


def test_phase_shifted_profile_is_flagged():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([3.0]), np.array([57.0]), np.array([2]), phi)
    assert asymmetry_metric(phi, target) > 0.5
    assert is_asymmetric(phi, target)


def test_phase_180_is_still_symmetric():
    """delta = 180 is representable in the cosine-only basis."""
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([180.0]), np.array([2]), phi)
    assert not is_asymmetric(phi, target)


def test_free_phase_fits_what_cosine_only_cannot():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([3.0]), np.array([57.0]), np.array([2]), phi)
    plain = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    free = fit_cosine_series(phi, target, [1, 2, 3, 4, 6], free_phase=True)
    assert free.rmse < 1e-8
    assert plain.rmse > free.rmse


def test_metric_handles_unsorted_angles():
    rng = np.random.default_rng(6)
    phi = rng.permutation(np.linspace(-180, 180, 73, endpoint=False))
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    assert asymmetry_metric(phi, target) < 1e-6
