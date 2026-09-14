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

ANGSTROM_TO_NM: float = 0.1
"""nm per Angstrom.  xtb and ORCA speak Angstrom; GROMACS and ffrefine speak nm."""

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


def angstrom_to_nm(x: Numeric) -> Numeric:
    """Convert Angstrom to nanometres."""
    return x * ANGSTROM_TO_NM


def nm_to_angstrom(x: Numeric) -> Numeric:
    """Convert nanometres to Angstrom."""
    return x / ANGSTROM_TO_NM


def deg_to_rad(x: Numeric) -> Numeric:
    """Convert degrees to radians."""
    return np.deg2rad(x)


def rad_to_deg(x: Numeric) -> Numeric:
    """Convert radians to degrees."""
    return np.rad2deg(x)
