"""Linear least-squares dihedral fitting (spec section 4.7).

The refit is exactly linear in a cosine basis, so no nonlinear optimiser is
needed.  Fit ``V = c + sum_n a_n cos(n phi)``, then convert:

    k_n = |a_n|,  delta_n = 0 if a_n > 0 else 180

With ``free_phase`` the basis gains sine terms and the conversion becomes
``k_n = hypot(a_n, b_n)``, ``delta_n = atan2(b_n, a_n)``, since

    k [1 + cos(n phi - delta)] = k + k cos(delta) cos(n phi) + k sin(delta) sin(n phi)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry import wrap_deg
from .basis import design_matrix


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
    coeffs, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)

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
    )
