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
from ..geometry import dihedral_deg_frames, kabsch_rmsd, wrap_deg

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


MAD_TO_SIGMA = 1.4826
"""Scales a median absolute deviation to a standard-deviation-equivalent for normal data."""

MIN_POINTS_FOR_OUTLIER = 4
"""Fewest adjacent gaps from which calling one an outlier is meaningful."""

UNIFORM_STEP_FACTOR = 3.0
"""When every step is the same size, a hop is one this many times the median step."""

MOTION_FLOOR_NM = 1e-6
"""Aligned RMSD below which a pair of frames counts as not having moved.

Kabsch superposition of two *identical* geometries does not return 0.0; it
returns the SVD's rounding residue, around 1e-17 nm.  Comparing a spread
against exact zero therefore tests floating-point luck rather than physics.  A
thousandth of a picometre sits eleven orders of magnitude above that residue
and four below any conformational change worth reporting.
"""


@dataclass
class Hop:
    """A suspected conformational hop between two adjacent scan frames.

    Attributes:
        index: the later frame of the pair (so the hop is between
            ``index - 1`` and ``index``, or between the last frame and frame 0
            when ``index`` is 0 and the scan is cyclic).
        phi_deg: that frame's measured dihedral.
        rmsd_nm: the aligned heavy-atom RMSD across the pair.
        threshold_nm: the outlier threshold it exceeded.
    """

    index: int
    phi_deg: float
    rmsd_nm: float
    threshold_nm: float


def find_hops(
    frames_nm: np.ndarray,
    heavy_indices: list[int],
    phi_deg: np.ndarray,
    k_mad: float = 5.0,
    cyclic: bool = True,
) -> list[Hop]:
    """Flag adjacent frames that look like a conformational hop (spec 4.3).

    **Why this is an outlier test and not a threshold.**  In a relaxed torsion
    scan, adjacent frames differ by the scanned dihedral *by construction*, so
    the aligned heavy-atom RMSD between them is never zero and no absolute
    cutoff carries meaning across molecules -- a large flexible molecule's
    normal step dwarfs a small rigid one's genuine hop.  What a hop actually
    looks like is a *spike* against the run of typical steps, so the statistic
    has to be relative to those steps and robust to the very outliers it is
    hunting.  Hence ``median + k * 1.4826 * MAD``.

    Hydrogens are excluded because a rotating methyl produces a large RMSD
    every step without any conformational change; spec 4.3 says heavy-atom
    RMSD for exactly this reason.

    The tool reports and continues (spec 4.3): a hop makes the profile
    unfittable by a smooth periodic function, but whether to re-scan, freeze
    another dihedral or accept it is the user's call, not this function's.

    Args:
        frames_nm: shape (nframes, natoms, 3).
        heavy_indices: 0-based indices of the non-hydrogen atoms.
        phi_deg: measured angles, shape (nframes,), for reporting.
        k_mad: how many scaled MADs above the median counts as a hop.
        cyclic: also compare the last frame with the first.  Correct for the
            default full-turn grid, where they are one step apart in angle;
            pass False for a partial scan, where they are not adjacent at all.

    Returns:
        The flagged hops, in frame order.  Empty when there are too few gaps
        to judge an outlier, when every gap is the same size, or when the
        frames never moved at all and the gaps are only rounding residue.
    """
    frames = np.asarray(frames_nm, float)
    if len(frames) < 2 or not heavy_indices:
        return []

    pairs = [(i - 1, i) for i in range(1, len(frames))]
    if cyclic and len(frames) > 2:
        pairs.append((len(frames) - 1, 0))

    rmsds = np.array(
        [kabsch_rmsd(frames[a][heavy_indices], frames[b][heavy_indices]) for a, b in pairs]
    )
    if len(rmsds) < MIN_POINTS_FOR_OUTLIER:
        return []

    median = float(np.median(rmsds))
    if median <= MOTION_FLOOR_NM:
        # Nothing moved: this whole run of RMSDs is superposition rounding
        # residue rather than geometry.  Running the outlier test over it would
        # always find a "hop" -- whichever pair happened to round up a few ulp
        # above the others -- and report it across a displacement of 1e-17 nm.
        return []

    mad = float(np.median(np.abs(rmsds - median)))
    spread = k_mad * MAD_TO_SIGMA * mad
    if spread > MOTION_FLOOR_NM:
        threshold = median + spread
    else:
        # Every adjacent step is the same size to within rounding, so the MAD
        # sets no scale.  That is the *easiest* baseline to judge against, not
        # an impossible one: bailing out here would blind the detector
        # precisely when the scan is cleanest, so a hop becomes any step
        # materially larger than the uniform one.
        threshold = median * UNIFORM_STEP_FACTOR

    hops = []
    for (_, later), rmsd in zip(pairs, rmsds):
        if rmsd > threshold:
            hops.append(
                Hop(
                    index=int(later),
                    phi_deg=float(phi_deg[later]),
                    rmsd_nm=float(rmsd),
                    threshold_nm=float(threshold),
                )
            )
    return sorted(hops, key=lambda h: h.index)
