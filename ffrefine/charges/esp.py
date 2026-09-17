"""Electrostatic potential arithmetic shared by both charge schemes.

Everything here is in atomic units -- coordinates in bohr, potentials in
Hartree per electron -- because that is what ``orca_vpot`` reads and writes,
and converting once at the process boundary beats converting in every formula.

Nothing in this module runs a subprocess, so the numerics can be tested
without ORCA installed.
"""

from __future__ import annotations

import numpy as np

from ..errors import ValidationError


def inverse_distances(atom_xyz_bohr: np.ndarray, grid_bohr: np.ndarray) -> np.ndarray:
    """``1 / |r_i - R_j|`` for every grid point i and atom j.

    Computed once and reused by :func:`potential_from_charges` and
    :func:`normal_equations`, so the two cannot disagree about which axis is
    which.

    Args:
        atom_xyz_bohr: shape (natoms, 3).
        grid_bohr: shape (npoints, 3).

    Returns:
        Shape ``(npoints, natoms)`` -- points first, matching the grid.

    Raises:
        ValidationError: a grid point coincides with an atom, which would make
            the entry infinite and hand the fit charges that still look finite.
    """
    atoms = np.asarray(atom_xyz_bohr, float)
    grid = np.asarray(grid_bohr, float)
    delta = grid[:, None, :] - atoms[None, :, :]
    r = np.sqrt((delta**2).sum(axis=-1))
    if not np.all(r > 1e-12):
        point, atom = np.unravel_index(int(np.argmin(r)), r.shape)
        raise ValidationError(
            f"grid point {point} coincides with atom {atom + 1} (separation "
            f"{r[point, atom]:.3e} bohr). The potential there is infinite, so the grid "
            "cannot be used; this usually means a vdW radius far too small for the element."
        )
    return 1.0 / r


def potential_from_charges(inv_r: np.ndarray, charges_e: np.ndarray) -> np.ndarray:
    """Potential at each grid point from a set of point charges."""
    return np.asarray(inv_r, float) @ np.asarray(charges_e, float)


def normal_equations(inv_r: np.ndarray, v_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The unrestrained least-squares system for charges against a potential.

    ``A_jk = sum_i 1/(r_ij r_ik)`` and ``B_j = sum_i V_i / r_ij``, which is the
    normal-equation form of minimising ``sum_i (V_i - sum_j q_j / r_ij)^2``.

    Returns:
        ``(A, B)`` with shapes ``(natoms, natoms)`` and ``(natoms,)``.
    """
    inv_r = np.asarray(inv_r, float)
    return inv_r.T @ inv_r, inv_r.T @ np.asarray(v_ref, float)


def rrms(v_ref: np.ndarray, v_fit: np.ndarray) -> float:
    """Relative root-mean-square deviation of a fitted potential.

    Relative rather than absolute because the potential's scale is set by the
    molecule and the grid, so an absolute RMS says nothing across systems.
    """
    ref = np.asarray(v_ref, float)
    denominator = float((ref**2).sum())
    if denominator <= 0.0:
        raise ValidationError("the reference potential is identically zero; RRMS is undefined")
    return float(np.sqrt(((ref - np.asarray(v_fit, float)) ** 2).sum() / denominator))
