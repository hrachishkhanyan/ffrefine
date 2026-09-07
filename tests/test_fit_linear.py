import logging

import numpy as np
import pytest
from ffrefine.fit.basis import design_matrix
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.mm.analytic import series_energy


def test_design_matrix_shape_and_constant_column():
    phi = np.linspace(-180, 180, 13)
    x = design_matrix(phi, [1, 2, 3])
    assert x.shape == (13, 4)
    assert np.allclose(x[:, 0], 1.0)


def test_design_matrix_free_phase_doubles_columns():
    phi = np.linspace(-180, 180, 13)
    x = design_matrix(phi, [1, 2, 3], free_phase=True)
    assert x.shape == (13, 7)


def test_recovers_single_term_exactly():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.5]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    got = dict(zip(r.mults.tolist(), r.k.tolist()))
    assert got[3] == pytest.approx(2.5, abs=1e-9)
    assert got[1] == pytest.approx(0.0, abs=1e-9)
    assert r.rmse == pytest.approx(0.0, abs=1e-9)


def test_recovers_phase_180_term():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([1.75]), np.array([180.0]), np.array([2]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    i = list(r.mults).index(2)
    assert r.k[i] == pytest.approx(1.75, abs=1e-9)
    assert r.delta_deg[i] == pytest.approx(180.0, abs=1e-6)


def test_recovers_multi_term_series_to_1e_minus_6():
    """This is the numerical core of the spec section 7.1 round-trip test."""
    phi = np.linspace(-180, 180, 73, endpoint=False)
    k_true = np.array([0.6, 1.2, 2.4])
    d_true = np.array([180.0, 0.0, 0.0])
    m_true = np.array([1, 2, 3])
    target = series_energy(k_true, d_true, m_true, phi)
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    for kt, dt, mt in zip(k_true, d_true, m_true):
        i = list(r.mults).index(mt)
        assert r.k[i] == pytest.approx(kt, abs=1e-6)
        assert r.delta_deg[i] == pytest.approx(dt, abs=1e-6)
    assert r.rmse == pytest.approx(0.0, abs=1e-9)


def test_unused_multiplicities_come_back_zero():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    for m in (1, 2, 4, 6):
        assert r.k[list(r.mults).index(m)] == pytest.approx(0.0, abs=1e-9)


def test_k_is_never_negative():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    rng = np.random.default_rng(3)
    target = rng.normal(size=phi.size)
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    assert np.all(r.k >= 0.0)


def test_constant_offset_is_absorbed_not_fitted():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    base = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    a = fit_cosine_series(phi, base, [1, 2, 3])
    b = fit_cosine_series(phi, base + 500.0, [1, 2, 3])
    assert a.k == pytest.approx(b.k, abs=1e-9)


def test_free_phase_recovers_arbitrary_phase():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([1.4]), np.array([57.0]), np.array([2]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3], free_phase=True)
    i = list(r.mults).index(2)
    assert r.k[i] == pytest.approx(1.4, abs=1e-8)
    assert r.delta_deg[i] == pytest.approx(57.0, abs=1e-6)


def test_weights_are_honoured():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    spoiled = target.copy()
    spoiled[0] += 1000.0
    w = np.ones_like(phi)
    w[0] = 0.0
    r = fit_cosine_series(phi, spoiled, [1, 2, 3], weights=w)
    assert r.k[list(r.mults).index(3)] == pytest.approx(2.0, abs=1e-6)


def test_residuals_and_predicted_are_consistent():
    phi = np.linspace(-180, 180, 37, endpoint=False)
    rng = np.random.default_rng(4)
    target = rng.normal(size=phi.size)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    assert r.residuals == pytest.approx(target - r.predicted, abs=1e-12)
    assert r.rmse == pytest.approx(np.sqrt(np.mean(r.residuals**2)), abs=1e-12)
    assert r.max_dev == pytest.approx(np.max(np.abs(r.residuals)), abs=1e-12)


def test_fit_records_the_design_matrix_rank():
    """Item 12: lstsq's rank was discarded."""
    phi = np.linspace(-180.0, 180.0, 24, endpoint=False)
    target = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    r = fit_cosine_series(phi, target, [1, 2, 3])
    assert r.rank == r.n_params == 4


def test_rank_deficiency_is_recorded_and_warned(caplog):
    """Two distinct points cannot determine a 4-parameter model: lstsq returns
    a minimum-norm solution that fits perfectly and means little."""
    phi = np.array([0.0, 90.0, 180.0, 270.0])
    target = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 0.0, 0.0])  # only two points survive
    # ffrefine.cli.setup_logging sets propagate=False on this logger, and test
    # order decides whether that has happened yet; force it so caplog sees the
    # record either way.
    logger = logging.getLogger("ffrefine")
    previous = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="ffrefine"):
            r = fit_cosine_series(phi, target, [1, 2, 3], weights=weights)
    finally:
        logger.propagate = previous
    assert r.rank < r.n_params
    assert r.rmse == pytest.approx(0.0, abs=1e-9)  # fits the survivors exactly
    assert "rank deficient" in caplog.text
