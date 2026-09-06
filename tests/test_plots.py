"""Tests for ffrefine.plots (spec sections 4.8, 4.9).

Plot code is easy to test in a way that verifies nothing (a blank canvas
satisfies ``path.exists()``). Beyond a handful of smoke tests, this file
checks the actual data attached to the drawn artists: which values ended up
on which line, and -- the one property here that would only otherwise be
caught by eyeballing a figure -- that angles are wrapped to (-180, 180] and
sorted before plotting, with every series carried through the *same*
permutation as phi so points don't get mismatched.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ffrefine import plots
from ffrefine.errors import ValidationError
from ffrefine.fit.linear import FitResult, fit_cosine_series
from ffrefine.mm.analytic import series_energy


@pytest.fixture
def fitted():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    return phi, target, fit_cosine_series(phi, target, [1, 2, 3])


@pytest.fixture
def captured(monkeypatch):
    """Record every (fig, ax) pair created via plt.subplots.

    plots.py closes each figure before its public function returns, so a
    caller has no other handle on it. plt.close() only deregisters the
    figure from pyplot's figure manager -- it does not clear the Figure
    object's artists -- so a reference grabbed before the close still lets
    us inspect ax.lines afterwards.
    """
    created: list[tuple] = []
    real_subplots = plots.plt.subplots

    def spy(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        created.append((fig, ax))
        return fig, ax

    monkeypatch.setattr(plots.plt, "subplots", spy)
    return created


def _fake_result(predicted, residuals=None, **overrides):
    predicted = np.asarray(predicted, dtype=float)
    fields = dict(
        mults=np.array([1, 2, 3]),
        k=np.array([0.0, 0.0, 0.0]),
        delta_deg=np.array([0.0, 0.0, 0.0]),
        constant=0.0,
        rmse=0.0,
        max_dev=0.0,
        residuals=np.zeros_like(predicted) if residuals is None else np.asarray(residuals, float),
        predicted=predicted,
        free_phase=False,
        n_params=4,
    )
    fields.update(overrides)
    return FitResult(**fields)


def _line_by_label_substring(ax, substring: str):
    """Find the one line whose label contains `substring` (case-insensitive)."""
    matches = [ln for ln in ax.lines if substring.lower() in ln.get_label().lower()]
    assert (
        len(matches) == 1
    ), f"expected exactly one line labelled like {substring!r}, found {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Basic smoke tests (from the brief): a file gets written for each function.
# Weak on their own (a blank canvas would also pass), kept only as a cheap
# "did it crash / did it produce output at all" check alongside the data
# assertions below.
# ---------------------------------------------------------------------------


def test_qm_profile_writes_a_file(tmp_path, fitted):
    phi, target, _ = fitted
    p = tmp_path / "qm.png"
    plots.plot_qm_profile(phi, target, p)
    assert p.exists() and p.stat().st_size > 1000


def test_mm_zeroed_plot_writes_a_file(tmp_path, fitted):
    phi, target, _ = fitted
    p = tmp_path / "mm.png"
    plots.plot_mm_zeroed(phi, target, p)
    assert p.exists() and p.stat().st_size > 1000


def test_fit_plot_writes_a_file(tmp_path, fitted):
    phi, target, r = fitted
    p = tmp_path / "fit.png"
    plots.plot_fit(phi, target, r, p)
    assert p.exists() and p.stat().st_size > 1000


def test_residual_plot_writes_a_file(tmp_path, fitted):
    phi, _, r = fitted
    p = tmp_path / "res.png"
    plots.plot_residuals(phi, r, p)
    assert p.exists() and p.stat().st_size > 1000


def test_validation_plot_writes_a_file(tmp_path, fitted):
    phi, target, r = fitted
    p = tmp_path / "val.png"
    plots.plot_validation(phi, target, target + 3.0, r.predicted, 3.0, 0.001, p)
    assert p.exists() and p.stat().st_size > 1000


def test_unsorted_angles_do_not_raise(tmp_path, fitted):
    phi, target, _ = fitted
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(phi))
    plots.plot_qm_profile(phi[perm], target[perm], tmp_path / "u.png")
    assert (tmp_path / "u.png").exists()


# ---------------------------------------------------------------------------
# The property that actually matters: wrap-then-sort, with every series
# carried through the *same* permutation as phi.
# ---------------------------------------------------------------------------

# Raw, deliberately unsorted and out-of-range angles. Each wraps to a
# distinct value in (-180, 180], chosen so the expected sort order is
# independently known and does not coincide with input order.
_RAW_PHI = np.array([200.0, -200.0, 100.0, -100.0, 0.0, 300.0, -300.0])
_EXPECTED_WRAPPED = np.array([-160.0, 160.0, 100.0, -100.0, 0.0, -60.0, 60.0])
_SORT_ORDER = np.argsort(_EXPECTED_WRAPPED, kind="stable")  # -> [0, 3, 5, 4, 6, 2, 1]


@pytest.mark.parametrize(
    "plot_fn", [plots.plot_qm_profile, plots.plot_mm_zeroed], ids=["qm_profile", "mm_zeroed"]
)
def test_single_series_plot_wraps_sorts_and_zeros_pairwise(tmp_path, captured, plot_fn):
    y_raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])

    plot_fn(_RAW_PHI, y_raw, tmp_path / "p.png")

    fig, ax = captured[0]
    assert len(ax.lines) == 1
    x, y = ax.lines[0].get_xydata().T

    expected_x = _EXPECTED_WRAPPED[_SORT_ORDER]
    expected_y = y_raw[_SORT_ORDER] - y_raw.min()
    assert x == pytest.approx(expected_x)
    assert y == pytest.approx(expected_y)

    # The specific property call out in the task: monotonically increasing,
    # wrapped into (-180, 180].
    assert np.all(np.diff(x) > 0)
    assert np.all((x > -180) & (x <= 180))


def test_minus_180_wraps_to_plus_180_not_left_at_the_boundary(tmp_path, captured):
    """wrap_deg's convention: -180 maps to +180, not left at -180 (spec 9.6)."""
    phi = np.array([-180.0, -90.0, 0.0])
    qm = np.array([1.0, 2.0, 3.0])

    plots.plot_qm_profile(phi, qm, tmp_path / "edge.png")

    fig, ax = captured[0]
    x, y = ax.lines[0].get_xydata().T
    # -180 -> +180 sorts last, ahead of nothing; -90 and 0 keep their values.
    assert x == pytest.approx([-90.0, 0.0, 180.0])
    assert y == pytest.approx([1.0, 2.0, 0.0])  # zeroed against qm.min() == 1.0


def test_fit_plot_puts_target_and_predicted_on_correctly_aligned_series(tmp_path, captured):
    target_raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    predicted_raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    result = _fake_result(predicted_raw)

    plots.plot_fit(_RAW_PHI, target_raw, result, tmp_path / "fit.png")

    fig, ax = captured[0]
    assert len(ax.lines) == 2
    expected_x = _EXPECTED_WRAPPED[_SORT_ORDER]

    target_line = _line_by_label_substring(ax, "target")
    fit_line = _line_by_label_substring(ax, "fit")
    assert target_line is not fit_line

    tx, ty = target_line.get_xydata().T
    fx, fy = fit_line.get_xydata().T
    assert tx == pytest.approx(expected_x)
    assert fx == pytest.approx(expected_x)
    assert ty == pytest.approx(target_raw[_SORT_ORDER])
    assert fy == pytest.approx(predicted_raw[_SORT_ORDER])
    # The two series must not be swapped.
    assert not np.allclose(ty, fy)


def test_residual_plot_shows_sorted_residuals_and_a_zero_reference(tmp_path, captured):
    residuals_raw = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    result = _fake_result(predicted=np.zeros_like(residuals_raw), residuals=residuals_raw)

    plots.plot_residuals(_RAW_PHI, result, tmp_path / "res.png")

    fig, ax = captured[0]
    assert len(ax.lines) == 2
    # axhline's Line2D carries exactly two points in a blended transform;
    # the data line carries one point per angle.
    data_lines = [ln for ln in ax.lines if len(ln.get_xdata()) == len(_RAW_PHI)]
    zero_lines = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]
    assert len(data_lines) == 1
    assert len(zero_lines) == 1

    x, y = data_lines[0].get_xydata().T
    expected_x = _EXPECTED_WRAPPED[_SORT_ORDER]
    assert x == pytest.approx(expected_x)
    assert y == pytest.approx(residuals_raw[_SORT_ORDER])


def test_validation_plot_zeroes_each_series_independently_and_aligns_them(tmp_path, captured):
    qm_raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    before_raw = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0])
    after_raw = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0])

    plots.plot_validation(_RAW_PHI, qm_raw, before_raw, after_raw, 3.0, 0.5, tmp_path / "val.png")

    fig, ax = captured[0]
    assert len(ax.lines) == 3
    expected_x = _EXPECTED_WRAPPED[_SORT_ORDER]

    qm_line = _line_by_label_substring(ax, "QM")
    before_line = _line_by_label_substring(ax, "before")
    after_line = _line_by_label_substring(ax, "after")

    for line, raw in ((qm_line, qm_raw), (before_line, before_raw), (after_line, after_raw)):
        x, y = line.get_xydata().T
        assert x == pytest.approx(expected_x)
        expected_y = raw[_SORT_ORDER] - raw.min()  # each series zeroed to its OWN minimum
        assert y == pytest.approx(expected_y)

    # RMSE values are surfaced somewhere in the labels, not silently dropped.
    assert "3.00" in before_line.get_label()
    assert "0.50" in after_line.get_label()


# ---------------------------------------------------------------------------
# Validation (ruling R29: bad input raises ValidationError, never ValueError)
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises_validation_error_not_value_error(tmp_path):
    phi = np.linspace(-180, 180, 10, endpoint=False)
    qm = np.zeros(9)
    with pytest.raises(ValidationError):
        plots.plot_qm_profile(phi, qm, tmp_path / "x.png")


def test_empty_input_raises_validation_error(tmp_path):
    with pytest.raises(ValidationError):
        plots.plot_qm_profile(np.array([]), np.array([]), tmp_path / "x.png")


def test_fit_plot_rejects_predicted_shape_mismatch(tmp_path, fitted):
    phi, target, r = fitted
    bad = _fake_result(predicted=r.predicted[:-1])
    with pytest.raises(ValidationError):
        plots.plot_fit(phi, target, bad, tmp_path / "x.png")


# ---------------------------------------------------------------------------
# Figures must not leak: the CLI draws several plots in one run.
# ---------------------------------------------------------------------------


def test_figures_are_closed_after_saving(tmp_path, fitted):
    phi, target, _ = fitted
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for i in range(25):
            plots.plot_qm_profile(phi, target, tmp_path / f"f{i}.png")
    assert plots.plt.get_fignums() == []


# ---------------------------------------------------------------------------
# Axis labelling sanity (not exact text, since that's an implementation
# choice, but the axes must actually be labelled).
# ---------------------------------------------------------------------------


def test_axes_are_labelled(tmp_path, captured, fitted):
    phi, target, _ = fitted
    plots.plot_qm_profile(phi, target, tmp_path / "qm.png")
    fig, ax = captured[0]
    assert "degree" in ax.get_xlabel().lower()
    assert ax.get_ylabel() != ""
    assert ax.get_title() != ""
