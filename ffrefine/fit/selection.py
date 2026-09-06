"""Multiplicity subset selection by BIC (spec section 4.7)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from .linear import FitResult, fit_cosine_series

DEFAULT_MULTIPLICITIES: tuple[int, ...] = (1, 2, 3, 4, 6)
"""Spec section 4.7; overridable with --multiplicities."""

DEFAULT_RMSE_TOLERANCE: float = 0.5
"""kJ/mol; a model within this of the best RMSE counts as good enough."""

_RMSE_FLOOR = 1e-12
"""Avoids log(0) for an exact fit."""


@dataclass
class Candidate:
    """One subset of multiplicities and how well it fitted."""

    mults: tuple[int, ...]
    result: FitResult | None
    bic: float
    rmse: float
    n_params: int


def bic(rmse: float, npoints: int, nparams: int) -> float:
    """Bayesian information criterion for a least-squares fit.

    ``n * ln(RSS/n) + k * ln(n)`` with ``RSS/n`` taken as ``rmse**2``.
    Lower is better.
    """
    variance = max(float(rmse) ** 2, _RMSE_FLOOR)
    return npoints * math.log(variance) + nparams * math.log(npoints)


def rank_candidates(
    phi_deg: np.ndarray,
    target_kj: np.ndarray,
    allowed_mults: Sequence[int] = DEFAULT_MULTIPLICITIES,
    weights: np.ndarray | None = None,
    free_phase: bool = False,
) -> list[Candidate]:
    """Fit every non-empty subset of the allowed multiplicities and rank by BIC.

    Args:
        phi_deg: measured dihedral values in degrees.
        target_kj: target difference curve in kJ/mol.
        allowed_mults: multiplicities to consider.
        weights: per-point weights; None means uniform.
        free_phase: allow arbitrary phases.

    Returns:
        Candidates sorted by ascending BIC.

    Raises:
        ValidationError: if ``allowed_mults`` is empty.
    """
    allowed = sorted(set(int(m) for m in allowed_mults))
    if not allowed:
        raise ValidationError("allowed_mults must not be empty")
    npoints = int(np.count_nonzero(weights)) if weights is not None else len(phi_deg)
    out: list[Candidate] = []
    for size in range(1, len(allowed) + 1):
        for subset in itertools.combinations(allowed, size):
            result = fit_cosine_series(
                phi_deg, target_kj, subset, weights=weights, free_phase=free_phase
            )
            out.append(
                Candidate(
                    mults=subset,
                    result=result,
                    bic=bic(result.rmse, npoints, result.n_params),
                    rmse=result.rmse,
                    n_params=result.n_params,
                )
            )
    out.sort(key=lambda c: c.bic)
    return out


def select_model(
    candidates: list[Candidate], rmse_tolerance: float = DEFAULT_RMSE_TOLERANCE
) -> Candidate:
    """Pick the smallest subset whose RMSE is within tolerance of the best.

    Args:
        candidates: as returned by :func:`rank_candidates`.
        rmse_tolerance: kJ/mol slack against the best RMSE achieved.

    Returns:
        The chosen candidate.  Ties on subset size are broken by BIC.

    Raises:
        ValidationError: if ``candidates`` is empty.
    """
    if not candidates:
        raise ValidationError("no candidates to select from")
    best_rmse = min(c.rmse for c in candidates)
    good = [c for c in candidates if c.rmse <= best_rmse + rmse_tolerance]
    return min(good, key=lambda c: (len(c.mults), c.bic))


def format_ranking_table(candidates: list[Candidate], limit: int = 20) -> str:
    """Render the ranking as a fixed-width table (spec section 4.7)."""
    lines = [
        f"{'multiplicities':<20} {'params':>6} {'RMSE kJ/mol':>12} {'BIC':>12}",
        "-" * 53,
    ]
    for c in candidates[:limit]:
        mults = ",".join(str(m) for m in c.mults)
        lines.append(f"{mults:<20} {c.n_params:>6d} {c.rmse:>12.4f} {c.bic:>12.2f}")
    if len(candidates) > limit:
        lines.append(f"... {len(candidates) - limit} further subsets not shown")
    return "\n".join(lines)
