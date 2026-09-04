"""Design matrices for the cosine fitting basis (spec section 4.7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def design_matrix(
    phi_deg: np.ndarray, mults: Sequence[int], free_phase: bool = False
) -> np.ndarray:
    """Build the linear design matrix for a dihedral cosine series.

    Columns are ``[1, cos(n1 phi), cos(n2 phi), ...]`` and, when ``free_phase``
    is set, the matching ``sin(n phi)`` columns.  The constant column absorbs
    the arbitrary energy offset and is discarded afterwards.

    Args:
        phi_deg: dihedral values in degrees, shape (npoints,).
        mults: multiplicities to include, e.g. (1, 2, 3, 4, 6).
        free_phase: add sine columns so arbitrary phases can be fitted.

    Returns:
        Shape (npoints, 1 + nmults) or (npoints, 1 + 2*nmults).
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    cols = [np.ones_like(phi)]
    cols += [np.cos(n * phi) for n in mults]
    if free_phase:
        cols += [np.sin(n * phi) for n in mults]
    return np.column_stack(cols)
