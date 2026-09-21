"""The Merz-Singh-Kollman shell grid the RESP fit is evaluated on.

Spec 5.2: four shells at 1.4, 1.6, 1.8 and 2.0 times the van der Waals radius,
about one point per square Angstrom, with points lying inside another atom's
shell discarded.

Two properties here are load-bearing rather than incidental.  The construction
is **deterministic** -- a golden spiral, never a random draw -- so two runs on
the same molecule give the same charges and these tests can assert exact
counts.  And an element with no tabulated radius is **refused** rather than
defaulted, because the radius sets where the fitting surface lies, so a guessed
one biases every charge in the molecule with nothing in the output to show it.

Nothing here runs a subprocess.
"""

from __future__ import annotations

import numpy as np

from ..errors import ValidationError
from ..units import ANGSTROM_TO_NM

MK_SHELL_SCALES: tuple[float, ...] = (1.4, 1.6, 1.8, 2.0)
"""Shell radii as multiples of the van der Waals radius (spec 5.2)."""

VDW_RADII_ANGSTROM: dict[str, float] = {
    "H": 1.20,
    "C": 1.50,
    "N": 1.50,
    "O": 1.40,
    "F": 1.35,
    "P": 1.80,
    "S": 1.75,
    "Cl": 1.70,
    "Br": 1.85,
    "I": 1.98,
}
"""Merz-Kollman radii, with Bondi values for Br and I, which MK does not define.

Recorded in ``charges.json`` so a later reader can tell which surface the
charges were fitted on.
"""


def vdw_radius_nm(element: str) -> float:
    """Van der Waals radius in nm.

    Raises:
        ValidationError: the element has no tabulated radius.  Substituting a
            neighbour's value would move the fitting surface and change every
            charge, silently, so this refuses instead.
    """
    symbol = element.strip().capitalize()
    if symbol not in VDW_RADII_ANGSTROM:
        raise ValidationError(
            f"no van der Waals radius for element {element!r}. The ESP grid is built from "
            f"these radii, so guessing one would bias every fitted charge. Known elements: "
            f"{', '.join(sorted(VDW_RADII_ANGSTROM))}."
        )
    return VDW_RADII_ANGSTROM[symbol] * ANGSTROM_TO_NM


def _unit_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform points on the unit sphere, by the golden spiral.

    Deterministic by construction.  The usual alternative -- drawing random
    directions until the count is met -- gives a different grid every run, so
    the same molecule would yield different charges each time and no test could
    pin the result.
    """
    if n < 1:
        return np.zeros((0, 3))
    index = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * index / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (1.0 + 5.0**0.5)
    theta = golden_angle * index
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def mk_grid(
    coords_nm: np.ndarray,
    elements: list[str],
    density_per_angstrom2: float = 1.0,
    scales: tuple[float, ...] = MK_SHELL_SCALES,
) -> np.ndarray:
    """Build the MK grid for one geometry.

    Args:
        coords_nm: shape (natoms, 3).
        elements: one symbol per atom, in the same order.
        density_per_angstrom2: points per square Angstrom of shell surface.
        scales: shell radii as multiples of the vdW radius.

    Returns:
        Shape ``(npoints, 3)`` in nm, in a deterministic order.

    Raises:
        ValidationError: the element list does not match the coordinates, an
            element has no radius, or every point was discarded.
    """
    coords = np.asarray(coords_nm, float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValidationError(f"coordinates have shape {coords.shape}, expected (natoms, 3)")
    if len(elements) != len(coords):
        raise ValidationError(
            f"{len(coords)} atoms but {len(elements)} element symbols; they must correspond"
        )
    if density_per_angstrom2 <= 0.0:
        raise ValidationError(f"--esp-density must be positive, got {density_per_angstrom2}")

    radii = np.array([vdw_radius_nm(e) for e in elements])
    kept: list[np.ndarray] = []
    for scale in scales:
        shell = radii * scale
        for i, centre in enumerate(coords):
            area_angstrom2 = 4.0 * np.pi * (shell[i] / ANGSTROM_TO_NM) ** 2
            n = int(round(area_angstrom2 * density_per_angstrom2))
            if n < 1:
                continue
            points = centre + shell[i] * _unit_sphere(n)
            # Discard points that fall inside any other atom's shell at this
            # scale: they sit in a region the potential is not being fitted on.
            distance = np.linalg.norm(points[:, None, :] - coords[None, :, :], axis=-1)
            inside = (distance < shell[None, :] - 1e-12).any(axis=1)
            kept.append(points[~inside])
    grid = np.vstack(kept) if kept else np.zeros((0, 3))
    if len(grid) == 0:
        raise ValidationError(
            "the ESP grid is empty: every shell point fell inside another atom. Check the "
            "geometry for overlapping atoms."
        )
    return grid
