"""Linear least-squares dihedral fitting (spec section 4.7).

The refit is exactly linear in a cosine basis, so no nonlinear optimiser is
needed.  Fit ``V = c + sum_n a_n cos(n phi)``, then convert:

    k_n = |a_n|,  delta_n = 0 if a_n > 0 else 180

With ``free_phase`` the basis gains sine terms and the conversion becomes
``k_n = hypot(a_n, b_n)``, ``delta_n = atan2(b_n, a_n)``, since

    k [1 + cos(n phi - delta)] = k + k cos(delta) cos(n phi) + k sin(delta) sin(n phi)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry import wrap_deg
from .basis import design_matrix

log = logging.getLogger("ffrefine")


@dataclass
class FitResult:
    """Outcome of one cosine-series fit.

    Attributes:
        mults: multiplicities in the model, ascending.
        k: force constants in kJ/mol, same order, all non-negative.
        delta_deg: phases in degrees, wrapped to (-180, 180].
        constant: the absorbed energy offset, discarded by consumers.
        rmse: weighted root-mean-square error in kJ/mol.
        max_dev: largest absolute residual in kJ/mol.
        residuals: target minus predicted, kJ/mol.
        predicted: fitted values at the input angles, kJ/mol.
        free_phase: whether sine terms were used.
        n_params: free parameters, for BIC.
        rank: the effective rank ``np.linalg.lstsq`` found for the weighted
            design matrix. Equal to ``n_params`` for a well-posed fit; less
            than that means the basis was not independent over the surviving
            points and ``lstsq`` returned a minimum-norm solution rather than
            a unique one -- the coefficients are then one arbitrary choice
            among infinitely many that fit equally well. Recorded rather than
            discarded because this project's signature failure is a
            plausible-looking number that is silently wrong.
    """

    mults: np.ndarray
    k: np.ndarray
    delta_deg: np.ndarray
    constant: float
    rmse: float
    max_dev: float
    residuals: np.ndarray
    predicted: np.ndarray
    free_phase: bool
    n_params: int
    rank: int = -1


def fit_cosine_series(
    phi_deg: np.ndarray,
    target_kj: np.ndarray,
    mults: Sequence[int],
    weights: np.ndarray | None = None,
    free_phase: bool = False,
) -> FitResult:
    """Fit a dihedral cosine series by weighted linear least squares.

    Args:
        phi_deg: measured dihedral values in degrees.
        target_kj: the target difference curve in kJ/mol.
        mults: multiplicities to include.
        weights: per-point weights; None means uniform.  Zero excludes a point.
        free_phase: allow arbitrary phases rather than 0/180 only.

    Returns:
        The fit, with k in kJ/mol and phases in degrees.
    """
    phi = np.asarray(phi_deg, dtype=float)
    y = np.asarray(target_kj, dtype=float)
    mults = list(mults)
    x = design_matrix(phi, mults, free_phase=free_phase)

    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)
    sw = np.sqrt(w)
    coeffs, _residues, rank, _sv = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
    rank = int(rank)
    if rank < x.shape[1]:
        # lstsq does not fail on a rank-deficient system: it silently returns
        # the minimum-norm solution, which fits just as well and means much
        # less. The CLI's `kept <= max_params` guard covers the realistic
        # triggers, so this is insurance -- but it is recorded in fit.json and
        # said out loud rather than thrown away.
        log.warning(
            "least-squares design matrix is rank deficient: rank %d for %d columns over "
            "%d weighted point(s). numpy returned a minimum-norm solution, so these "
            "coefficients are one arbitrary choice among many that fit equally well.",
            rank,
            x.shape[1],
            int(np.count_nonzero(w)),
        )

    constant = float(coeffs[0])
    n = len(mults)
    a = coeffs[1 : 1 + n]
    if free_phase:
        b = coeffs[1 + n : 1 + 2 * n]
        k = np.hypot(a, b)
        delta = wrap_deg(np.degrees(np.arctan2(b, a)))
    else:
        k = np.abs(a)
        delta = np.where(a >= 0.0, 0.0, 180.0)

    predicted = x @ coeffs
    residuals = y - predicted
    nz = w > 0
    rmse = float(np.sqrt(np.average(residuals[nz] ** 2, weights=w[nz]))) if nz.any() else 0.0
    max_dev = float(np.max(np.abs(residuals[nz]))) if nz.any() else 0.0

    return FitResult(
        mults=np.array(mults, dtype=int),
        k=np.asarray(k, dtype=float),
        delta_deg=np.asarray(delta, dtype=float),
        constant=constant,
        rmse=rmse,
        max_dev=max_dev,
        residuals=residuals,
        predicted=predicted,
        free_phase=free_phase,
        n_params=int(coeffs.size),
        rank=rank,
    )


DEFAULT_ASYMMETRY_TOLERANCE: float = 0.5
"""kJ/mol; above this the cosine-only basis cannot represent the profile."""


def asymmetry_metric(phi_deg: np.ndarray, target_kj: np.ndarray) -> float:
    """RMS mismatch between a torsion profile and its mirror about phi = 0.

    A cosine-only basis is even in phi, so it can only represent profiles
    symmetric about zero.  Chiral or otherwise phase-shifted torsions break
    that assumption (spec section 4.7).

    Measured angles are not on an exact grid, so the mirrored profile is
    obtained by periodic interpolation rather than by index reversal.

    Args:
        phi_deg: measured dihedral values in degrees, any order.
        target_kj: the profile in kJ/mol.

    Returns:
        RMS difference in kJ/mol between the profile and its mirror.
    """
    phi = wrap_deg(np.asarray(phi_deg, dtype=float))
    y = np.asarray(target_kj, dtype=float)
    order = np.argsort(phi)
    ps, ys = phi[order], y[order]
    # Extend periodically so interpolation at -phi never falls off the ends.
    ext_p = np.concatenate([ps - 360.0, ps, ps + 360.0])
    ext_y = np.concatenate([ys, ys, ys])
    mirrored = np.interp(-ps, ext_p, ext_y)
    return float(np.sqrt(np.mean((ys - mirrored) ** 2)))


def is_asymmetric(
    phi_deg: np.ndarray,
    target_kj: np.ndarray,
    tolerance: float = DEFAULT_ASYMMETRY_TOLERANCE,
) -> bool:
    """Whether a profile is too asymmetric for the cosine-only basis."""
    return asymmetry_metric(phi_deg, target_kj) > tolerance
