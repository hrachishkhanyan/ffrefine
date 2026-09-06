"""Diagnostic figures (spec sections 4.8 and 4.9).

The Agg backend is selected at import so plotting works with no display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .errors import ValidationError  # noqa: E402
from .fit.linear import FitResult  # noqa: E402
from .geometry import wrap_deg  # noqa: E402

_XLABEL = "dihedral / degrees"
_YLABEL = "energy / kJ mol$^{-1}$"
_TICKS = [-180, -120, -60, 0, 60, 120, 180]


def _sorted(phi_deg: np.ndarray, *series: np.ndarray) -> tuple[np.ndarray, ...]:
    """Wrap and sort by angle so plotted lines do not jump at the period edge.

    A stable sort is used so that points sharing a wrapped angle keep their
    relative input order, making plotted output deterministic.

    Args:
        phi_deg: dihedral values in degrees, any order or range.
        *series: zero or more arrays, each the same shape as phi_deg, to be
            carried through the same wrap-and-sort permutation.

    Returns:
        A tuple ``(phi_wrapped_sorted, *series_sorted)``.

    Raises:
        ValidationError: if phi_deg is empty, or any series' shape does not
            match phi_deg's shape.
    """
    phi_arr = np.asarray(phi_deg, dtype=float)
    if phi_arr.size == 0:
        raise ValidationError("phi_deg must not be empty")
    arrs = [np.asarray(s, dtype=float) for s in series]
    for a in arrs:
        if a.shape != phi_arr.shape:
            raise ValidationError(f"series shape mismatch: {a.shape} vs phi_deg {phi_arr.shape}")
    phi = wrap_deg(phi_arr)
    order = np.argsort(phi, kind="stable")
    return (phi[order], *(a[order] for a in arrs))


def _finish(ax, path: str | Path, title: str) -> None:
    """Apply shared axis styling, save the figure, then close it.

    Closing after every save keeps the module usable in long-running CLI
    invocations that draw many plots without accumulating open figures.
    """
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)
    ax.set_title(title)
    ax.set_xticks(_TICKS)
    ax.set_xlim(-180, 180)
    ax.grid(alpha=0.3)
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_qm_profile(phi_deg: np.ndarray, qm_kj: np.ndarray, path: str | Path) -> None:
    """Relaxed QM torsion profile, shifted to its own minimum (spec 4.8)."""
    phi, qm = _sorted(phi_deg, qm_kj)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, qm - qm.min(), "o-", color="#1f77b4", label="QM")
    _finish(ax, path, "QM torsion profile")


def plot_mm_zeroed(phi_deg: np.ndarray, mm_kj: np.ndarray, path: str | Path) -> None:
    """MM profile with the target torsion's own contribution removed (spec 4.8)."""
    phi, mm = _sorted(phi_deg, mm_kj)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, mm - mm.min(), "s-", color="#d62728", label="MM, torsion zeroed")
    _finish(ax, path, "MM profile without the target torsion")


def plot_fit(
    phi_deg: np.ndarray, target_kj: np.ndarray, result: FitResult, path: str | Path
) -> None:
    """Target difference curve with the fitted cosine series overlaid (spec 4.8)."""
    phi, target, predicted = _sorted(phi_deg, target_kj, result.predicted)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, target, "o", color="#1f77b4", label="target (QM - MM$_0$)")
    ax.plot(phi, predicted, "-", color="#ff7f0e", linewidth=2, label="fitted series")
    terms = ", ".join(f"n={m}" for m in result.mults)
    _finish(ax, path, f"Dihedral fit ({terms}), RMSE {result.rmse:.3f} kJ/mol")


def plot_residuals(phi_deg: np.ndarray, result: FitResult, path: str | Path) -> None:
    """Fit residuals against dihedral angle (spec 4.8)."""
    phi, res = _sorted(phi_deg, result.residuals)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0.0, color="#888888", linewidth=1)
    ax.plot(phi, res, "o-", color="#2ca02c", label="residual")
    _finish(ax, path, f"Residuals, max |dev| {result.max_dev:.3f} kJ/mol")


def plot_validation(
    phi_deg: np.ndarray,
    qm_kj: np.ndarray,
    mm_before_kj: np.ndarray,
    mm_after_kj: np.ndarray,
    rmse_before: float,
    rmse_after: float,
    path: str | Path,
) -> None:
    """Total MM energy against QM, before and after refitting (spec section 4.9)."""
    phi, qm, before, after = _sorted(phi_deg, qm_kj, mm_before_kj, mm_after_kj)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(phi, qm - qm.min(), "o-", color="#1f77b4", label="QM")
    ax.plot(
        phi,
        before - before.min(),
        "s--",
        color="#d62728",
        label=f"MM before (RMSE {rmse_before:.2f})",
    )
    ax.plot(
        phi,
        after - after.min(),
        "^-",
        color="#2ca02c",
        label=f"MM after (RMSE {rmse_after:.2f})",
    )
    _finish(ax, path, "Validation: MM against QM before and after refitting")
