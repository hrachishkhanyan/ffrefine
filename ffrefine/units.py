"""The single place where unit conversions live (spec section 9.8).

GROMACS works in kJ/mol, CHARMM .prm files in kcal/mol, xtb and ORCA in
Hartree.  No other module in ffrefine may define a conversion constant.
"""

from __future__ import annotations

import numpy as np

HARTREE_TO_KJ: float = 2625.499639
"""kJ/mol per Hartree."""

KCAL_TO_KJ: float = 4.184
"""kJ per thermochemical kcal."""

Numeric = float | np.ndarray


def hartree_to_kj(x: Numeric) -> Numeric:
    """Convert Hartree to kJ/mol."""
    return x * HARTREE_TO_KJ


def kj_to_hartree(x: Numeric) -> Numeric:
    """Convert kJ/mol to Hartree."""
    return x / HARTREE_TO_KJ


def kcal_to_kj(x: Numeric) -> Numeric:
    """Convert kcal/mol to kJ/mol."""
    return x * KCAL_TO_KJ


def kj_to_kcal(x: Numeric) -> Numeric:
    """Convert kJ/mol to kcal/mol."""
    return x / KCAL_TO_KJ


def deg_to_rad(x: Numeric) -> Numeric:
    """Convert degrees to radians."""
    return np.deg2rad(x)


def rad_to_deg(x: Numeric) -> Numeric:
    """Convert radians to degrees."""
    return np.rad2deg(x)
