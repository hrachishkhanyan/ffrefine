"""Analysis of xtb scan output: angles, bidirectional merge, discontinuities.

Nothing in this module runs a subprocess.  That is deliberate: every rule it
implements -- which forward point pairs with which reverse point, what counts
as a conformational hop, when a hydrogen bond has formed -- is a judgement
that decides whether a published torsion profile is trustworthy, and those
judgements are testable only if they can be exercised on captured and
synthetic data without an external tool in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import dihedral_deg_frames, wrap_deg

Quad = tuple[int, int, int, int]


def measure_angles(frames_nm: np.ndarray, quad: Quad) -> np.ndarray:
    """Measure the target dihedral in every frame, in degrees.

    Spec 9.5: "Measure the actual dihedral value from each frame's
    coordinates. Do not trust the requested grid."  This is not a formality --
    on the committed 4-point butane fixture, a scan requested at 0/120/240/360
    came back at 180/120/-120/0, so code that labelled points by the request
    would report the anti minimum as the syn barrier.

    Args:
        frames_nm: shape (nframes, natoms, 3).
        quad: 1-based atom indices, as the user and the topology write them.

    Returns:
        Shape (nframes,) in degrees, wrapped to (-180, 180].
    """
    zero_based = tuple(i - 1 for i in quad)
    return dihedral_deg_frames(np.asarray(frames_nm, float), zero_based)


@dataclass
class ScanLeg:
    """One direction of a (possibly bidirectional) relaxed scan.

    Attributes:
        frames_nm: shape (nframes, natoms, 3) in nanometres.
        energy_kj: shape (nframes,) absolute energies in kJ/mol.
        phi_deg: shape (nframes,) measured target dihedral, degrees.
        direction: ``"forward"`` or ``"reverse"``, for messages and provenance.
    """

    frames_nm: np.ndarray
    energy_kj: np.ndarray
    phi_deg: np.ndarray
    direction: str

    def __post_init__(self) -> None:
        n = len(self.energy_kj)
        if len(self.phi_deg) != n or len(self.frames_nm) != n:
            raise ValueError(
                f"ScanLeg arrays disagree: {len(self.frames_nm)} frames, {n} energies, "
                f"{len(self.phi_deg)} angles"
            )

    def __len__(self) -> int:
        return len(self.energy_kj)
