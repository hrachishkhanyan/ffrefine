"""Point weighting for the dihedral fit (spec section 4.7)."""

from __future__ import annotations

import numpy as np

GAS_CONSTANT_KJ: float = 0.0083144626
"""Molar gas constant in kJ/mol/K."""

DEFAULT_EMAX: float = 40.0
"""kJ/mol above the QM minimum; points higher than this are discarded."""

DEFAULT_TEMPERATURE: float = 300.0
"""Kelvin, for Boltzmann weighting."""


def _relative(qm_kj: np.ndarray) -> np.ndarray:
    qm = np.asarray(qm_kj, dtype=float)
    return qm - qm.min()


def uniform_weights(qm_kj: np.ndarray, emax: float | None = DEFAULT_EMAX) -> np.ndarray:
    """Uniform weights, discarding points high above the QM minimum.

    Args:
        qm_kj: QM energies in kJ/mol; only differences matter.
        emax: cut-off in kJ/mol above the minimum, or None to keep everything.

    Returns:
        Weights of 1.0 (kept) or 0.0 (discarded).
    """
    rel = _relative(qm_kj)
    if emax is None:
        return np.ones_like(rel)
    return (rel <= emax).astype(float)


def boltzmann_weights(
    qm_kj: np.ndarray,
    temperature: float = DEFAULT_TEMPERATURE,
    emax: float | None = DEFAULT_EMAX,
) -> np.ndarray:
    """Boltzmann weights ``exp(-E/RT)`` relative to the QM minimum.

    Normalised so the lowest-energy point has weight 1.  The ``emax`` cut is
    still applied on top, so a Boltzmann fit never silently includes points the
    uniform scheme would have thrown away.

    Args:
        qm_kj: QM energies in kJ/mol.
        temperature: in Kelvin.
        emax: cut-off in kJ/mol above the minimum, or None.

    Returns:
        Weights in [0, 1].
    """
    rel = _relative(qm_kj)
    out = np.exp(-rel / (GAS_CONSTANT_KJ * temperature))
    if emax is not None:
        out = np.where(rel <= emax, out, 0.0)
    return out
