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

from ..errors import ValidationError
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


@dataclass
class MergeResult:
    """The profile produced by combining a forward and a reverse leg.

    Attributes:
        frames_nm: chosen geometry per grid point, shape (npoints, natoms, 3).
        energy_kj: chosen energy per grid point, shape (npoints,).
        phi_deg: the forward leg's measured angles, shape (npoints,).
        source: ``"forward"`` or ``"reverse"`` per point, saying which leg won.
        max_hysteresis_kj: largest |E_forward - E_reverse| over paired points.
        worst_angle_deg: the angle where that maximum occurred, or nan.
        unpaired_forward_deg: forward angles with no reverse partner in tolerance.
        unpaired_reverse_deg: reverse angles that partnered nothing.
    """

    frames_nm: np.ndarray
    energy_kj: np.ndarray
    phi_deg: np.ndarray
    source: list[str]
    max_hysteresis_kj: float
    worst_angle_deg: float
    unpaired_forward_deg: list[float]
    unpaired_reverse_deg: list[float]


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute separation between two angles in degrees.

    Wrap-aware: -179 and +179 are two degrees apart, not 358.
    """
    return np.abs(wrap_deg(np.asarray(a, float) - np.asarray(b, float)))


def merge_bidirectional(forward: ScanLeg, reverse: ScanLeg, tol_deg: float) -> MergeResult:
    """Combine two legs, taking the lower energy at each grid point (spec 4.3).

    **Points pair by measured angle, never by array index.**  Two independent
    relaxed scans do not land on the same angles, and the committed 4-point
    butane fixtures show why this matters: the forward leg measured
    180/120/-120/0 while the reverse measured 0/-120/120/0.  Index pairing
    would pair 180 with 0 -- the anti minimum against the syn barrier -- and
    "take the lower energy" would then write the minimum onto the syn point,
    erasing the 20.7 kJ/mol barrier the scan exists to measure.  Three of the
    four index pairings are wrong, and reversed-index pairing fares no better:
    the reverse leg visits 0 twice and never reaches 180 at all.

    The output grid is the forward leg's, because that is the grid the user
    asked for.  A reverse point matching nothing is *reported*, not merged in,
    which would otherwise produce duplicated abscissae and an uneven grid.

    Args:
        forward: the leg run from the input geometry.
        reverse: the leg run back from the forward leg's final structure.
        tol_deg: largest angular separation still considered the same grid
            point.  Half a step is the sensible value.

    Returns:
        A :class:`MergeResult`.

    Raises:
        ValidationError: two forward points claim the same reverse point,
            which means the grid is degenerate and the intended pairing is
            genuinely ambiguous.
    """
    frames = np.array(forward.frames_nm, dtype=float, copy=True)
    energies = np.array(forward.energy_kj, dtype=float, copy=True)
    source = ["forward"] * len(forward)
    claimed: dict[int, int] = {}
    hysteresis = 0.0
    worst_angle = float("nan")
    unpaired_forward: list[float] = []

    for i, phi in enumerate(forward.phi_deg):
        if len(reverse) == 0:
            unpaired_forward.append(float(phi))
            continue
        separation = angular_distance_deg(reverse.phi_deg, phi)
        j = int(np.argmin(separation))
        if separation[j] > tol_deg:
            unpaired_forward.append(float(phi))
            continue
        if j in claimed:
            raise ValidationError(
                f"forward points at {forward.phi_deg[claimed[j]]:.2f} and {phi:.2f} degrees "
                f"both pair with the reverse point at {reverse.phi_deg[j]:.2f} degrees "
                f"within the {tol_deg:g} degree tolerance. The grid is degenerate, so "
                "which pairing was intended is ambiguous; re-run with a coarser --step "
                "or a tighter tolerance."
            )
        claimed[j] = i

        gap = abs(float(forward.energy_kj[i] - reverse.energy_kj[j]))
        if gap > hysteresis:
            hysteresis = gap
            worst_angle = float(phi)
        if reverse.energy_kj[j] < forward.energy_kj[i]:
            energies[i] = reverse.energy_kj[j]
            frames[i] = reverse.frames_nm[j]
            source[i] = "reverse"

    unpaired_reverse = [float(reverse.phi_deg[j]) for j in range(len(reverse)) if j not in claimed]
    return MergeResult(
        frames_nm=frames,
        energy_kj=energies,
        phi_deg=np.array(forward.phi_deg, dtype=float, copy=True),
        source=source,
        max_hysteresis_kj=hysteresis,
        worst_angle_deg=worst_angle,
        unpaired_forward_deg=unpaired_forward,
        unpaired_reverse_deg=unpaired_reverse,
    )
