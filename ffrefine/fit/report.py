"""Fit statistics: minima, barriers, and the fit.json payload (spec 4.7, 4.8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from ..geometry import wrap_deg
from ..units import kj_to_kcal
from .linear import FitResult, asymmetry_metric


@dataclass
class ProfileStats:
    """Shape of a torsion profile.

    Attributes:
        minima_deg: angular positions of local minima, degrees, (-180, 180].
        minima_kj: energy at each minimum, kJ/mol, same order as minima_deg.
        barriers_kj: barrier height between each pair of consecutive minima
            (wrapping around the periodic boundary), kJ/mol.
        global_min_deg: angular position of the lowest-energy minimum.
        max_barrier_kj: the largest entry in barriers_kj, or 0.0 if there is
            at most one minimum.
    """

    minima_deg: list[float]
    minima_kj: list[float]
    barriers_kj: list[float]
    global_min_deg: float
    max_barrier_kj: float


def profile_stats(phi_deg: np.ndarray, energy_kj: np.ndarray) -> ProfileStats:
    """Locate minima and barrier heights of a periodic torsion profile.

    The profile is periodic in 360 degrees, so the neighbour comparison used
    to detect minima wraps around the array ends: a minimum sitting at or
    near +/-180 is compared against the samples on the far side of the wrap
    and is not missed by an interior-only scan.

    Args:
        phi_deg: dihedral values in degrees, any order.
        energy_kj: energies in kJ/mol, same shape as phi_deg.

    Returns:
        Minima positions and depths, barrier heights between consecutive
        minima, and the global minimum.

    Raises:
        ValidationError: if the inputs are empty or their shapes mismatch.
    """
    phi_arr = np.asarray(phi_deg, dtype=float)
    e_arr = np.asarray(energy_kj, dtype=float)
    if phi_arr.shape != e_arr.shape:
        raise ValidationError(
            f"phi_deg and energy_kj shape mismatch: {phi_arr.shape} vs {e_arr.shape}"
        )
    if phi_arr.size == 0:
        raise ValidationError("phi_deg and energy_kj must not be empty")

    phi = wrap_deg(phi_arr)
    e = e_arr
    order = np.argsort(phi)
    p, y = phi[order], e[order]

    prev = np.roll(y, 1)
    nxt = np.roll(y, -1)
    is_min = (y < prev) & (y <= nxt)
    minima_idx = np.flatnonzero(is_min)

    barriers: list[float] = []
    for a, b in zip(minima_idx, np.roll(minima_idx, -1)):
        if a < b:
            segment = y[a : b + 1]
        else:  # wraps through the period boundary (also covers a single minimum)
            segment = np.concatenate([y[a:], y[: b + 1]])
        barriers.append(float(segment.max() - min(y[a], y[b])))

    global_idx = int(np.argmin(y))
    return ProfileStats(
        minima_deg=[float(p[i]) for i in minima_idx],
        minima_kj=[float(y[i]) for i in minima_idx],
        barriers_kj=barriers,
        global_min_deg=float(p[global_idx]),
        max_barrier_kj=float(max(barriers)) if barriers else 0.0,
    )


def fit_summary(
    result: FitResult,
    phi_deg: np.ndarray,
    target_kj: np.ndarray,
    provenance: dict | None = None,
) -> dict:
    """Build the JSON-serialisable fit record written to fit.json (spec 4.8).

    Energies are reported in kJ/mol, with kcal/mol alongside for users going
    back to CHARMM.  Conversion goes through ``units`` (spec section 9.8).
    Every value is converted to a builtin ``float``/``int``/``list`` -- numpy
    scalars and arrays are not accepted by the stdlib ``json`` module.

    Args:
        result: the fit to summarise.
        phi_deg: dihedral values the fit was evaluated at, degrees.
        target_kj: the target curve the fit was made against, kJ/mol.
        provenance: what produced this fit -- the target quadruplet, the
            geometry digest binding it to a particular ``scan.json``, and the
            settings that shaped it (spec 4.8's "every number"). Recorded
            verbatim under ``"provenance"``. ``validate`` reads
            ``quad_1_based`` and ``frames_sha256`` back out of it to refuse
            applying this fit to a different torsion or a different scan.

    Returns:
        A dict safe to pass to ``json.dumps``.

    Raises:
        ValidationError: if phi_deg and target_kj shapes mismatch, or either
            mismatches the length of the fit's own arrays.
    """
    phi_arr = np.asarray(phi_deg, dtype=float)
    target_arr = np.asarray(target_kj, dtype=float)
    if phi_arr.shape != target_arr.shape:
        raise ValidationError(
            f"phi_deg and target_kj shape mismatch: {phi_arr.shape} vs {target_arr.shape}"
        )
    if phi_arr.shape != np.asarray(result.predicted).shape:
        raise ValidationError(
            "phi_deg does not match the fit: "
            f"{phi_arr.shape} vs predicted {np.asarray(result.predicted).shape}"
        )

    stats = profile_stats(phi_arr, result.predicted)
    return {
        "provenance": dict(provenance or {}),
        "multiplicities": [int(m) for m in result.mults],
        "terms": [
            {
                "multiplicity": int(m),
                "k_kj_per_mol": float(k),
                "k_kcal_per_mol": float(kj_to_kcal(k)),
                "phase_deg": float(d),
            }
            for m, k, d in zip(result.mults, result.k, result.delta_deg)
        ],
        "free_phase": bool(result.free_phase),
        "rmse_kj_per_mol": float(result.rmse),
        "max_deviation_kj_per_mol": float(result.max_dev),
        "asymmetry_kj_per_mol": float(asymmetry_metric(phi_arr, target_arr)),
        "n_points": int(phi_arr.size),
        "n_parameters": int(result.n_params),
        "phi_deg": [float(v) for v in phi_arr],
        "target_kj_per_mol": [float(v) for v in target_arr],
        "fitted_kj_per_mol": [float(v) for v in result.predicted],
        "residuals_kj_per_mol": [float(v) for v in result.residuals],
        "profile": {
            "minima_deg": stats.minima_deg,
            "minima_kj_per_mol": stats.minima_kj,
            "barriers_kj_per_mol": stats.barriers_kj,
            "global_minimum_deg": stats.global_min_deg,
            "max_barrier_kj_per_mol": stats.max_barrier_kj,
        },
    }
