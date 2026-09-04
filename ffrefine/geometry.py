"""Geometric primitives: dihedral measurement, angle wrapping, RMSD.

All atom indices taken by this module are 0-based; callers convert from the
user-facing 1-based indices at the CLI boundary.
"""

from __future__ import annotations

import numpy as np

Numeric = float | np.ndarray


def wrap_deg(a: Numeric) -> Numeric:
    """Wrap an angle in degrees to the half-open interval (-180, 180].

    -180 maps to +180 so the convention is consistent across the package
    (spec section 9.6).
    """
    return 180.0 - np.mod(180.0 - np.asarray(a, dtype=float), 360.0)


def dihedral_deg(p0, p1, p2, p3) -> float:
    """Signed IUPAC dihedral angle p0-p1-p2-p3 in degrees, wrapped to (-180, 180].

    Uses the numerically stable atan2 formulation. Unit-agnostic.
    """
    b0 = np.asarray(p0, float) - np.asarray(p1, float)
    b1 = np.asarray(p2, float) - np.asarray(p1, float)
    b2 = np.asarray(p3, float) - np.asarray(p2, float)
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(wrap_deg(np.degrees(np.arctan2(y, x))))


def dihedral_deg_frames(coords: np.ndarray, idx: tuple[int, int, int, int]) -> np.ndarray:
    """Measure one dihedral across many frames.

    Args:
        coords: shape (nframes, natoms, 3).
        idx: 0-based atom indices (i, j, k, l).

    Returns:
        Array of shape (nframes,) in degrees, wrapped to (-180, 180].
    """
    coords = np.asarray(coords, float)
    i, j, k, l = idx
    b0 = coords[:, i] - coords[:, j]
    b1 = coords[:, k] - coords[:, j]
    b2 = coords[:, l] - coords[:, k]
    b1n = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - (np.sum(b0 * b1n, axis=1)[:, None]) * b1n
    w = b2 - (np.sum(b2 * b1n, axis=1)[:, None]) * b1n
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1n, v) * w, axis=1)
    return wrap_deg(np.degrees(np.arctan2(y, x)))


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two coordinate sets after optimal rigid superposition.

    Args:
        a, b: shape (n, 3), same atom ordering.

    Returns:
        RMSD in the input distance unit.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    cov = ac.T @ bc
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = ac @ rot.T - bc
    return float(np.sqrt((diff**2).sum() / len(a)))
