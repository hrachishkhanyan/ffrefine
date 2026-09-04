"""Analytic evaluation of GROMACS proper-dihedral terms.

This is the independent reference used by the mandatory self-consistency check
(spec section 4.5).  It is derived from the functional form alone and must
never be tuned to reproduce GROMACS output -- that would defeat the check.

Functional form, GROMACS funct 9 (and funct 1):

    V(phi) = k * [1 + cos(n * phi - phi0)]

with phi0 in degrees, k in kJ/mol and n a non-negative integer.
"""

from __future__ import annotations

import numpy as np

from ..geometry import dihedral_deg_frames
from ..io.topology import Dihedral


def dihedral_energy(params: list[float], phi_deg: np.ndarray) -> np.ndarray:
    """Energy of a single funct-9 dihedral term in kJ/mol.

    Args:
        params: ``[phi0_deg, k_kj_per_mol, multiplicity]``.
        phi_deg: dihedral values in degrees; any range, the form is periodic.

    Returns:
        Energies in kJ/mol, same shape as ``phi_deg``.
    """
    phi0, k, mult = float(params[0]), float(params[1]), int(round(float(params[2])))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return k * (1.0 + np.cos(mult * phi - np.deg2rad(phi0)))


def torsion_set_energy(dihedrals: list[Dihedral], coords: np.ndarray) -> np.ndarray:
    """Total energy of a set of dihedral entries across frames.

    Entries with no explicit parameters contribute zero: their parameters live
    in [ dihedraltypes ] and cannot be evaluated without the force field, which
    is exactly why the section 4.5 check compares against GROMACS instead.

    Args:
        dihedrals: entries with **1-based** atom indices.
        coords: shape (nframes, natoms, 3).

    Returns:
        Array of shape (nframes,) in kJ/mol.
    """
    coords = np.asarray(coords, dtype=float)
    total = np.zeros(coords.shape[0], dtype=float)
    for d in dihedrals:
        if not d.params:
            continue
        idx = (d.ai - 1, d.aj - 1, d.ak - 1, d.al - 1)  # 1-based -> 0-based
        phi = dihedral_deg_frames(coords, idx)
        total += dihedral_energy(d.params, phi)
    return total


def series_energy(
    k: np.ndarray, delta_deg: np.ndarray, mult: np.ndarray, phi_deg: np.ndarray
) -> np.ndarray:
    """Evaluate a fitted cosine series sum_n k_n [1 + cos(n*phi - delta_n)].

    Args:
        k: force constants in kJ/mol, shape (nterms,).
        delta_deg: phases in degrees, shape (nterms,).
        mult: multiplicities, shape (nterms,).
        phi_deg: dihedral values in degrees.

    Returns:
        Energies in kJ/mol, same shape as ``phi_deg``.
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    out = np.zeros_like(phi, dtype=float)
    for kn, dn, n in zip(np.atleast_1d(k), np.atleast_1d(delta_deg), np.atleast_1d(mult)):
        out += kn * (1.0 + np.cos(int(n) * phi - np.deg2rad(dn)))
    return out
