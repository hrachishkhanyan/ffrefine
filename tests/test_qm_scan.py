import numpy as np
import pytest
from pathlib import Path

from ffrefine.errors import ValidationError
from ffrefine.qm.scan import Hop, ScanLeg, find_hops, measure_angles, merge_bidirectional
from ffrefine.qm.xtb import parse_scan_log

FIXTURES = Path(__file__).parent.parent / "fixtures" / "butane" / "xtb"


def _leg(path: Path, direction: str) -> ScanLeg:
    frames_nm, energy_kj = parse_scan_log(path)
    return ScanLeg(frames_nm, energy_kj, measure_angles(frames_nm, (1, 5, 8, 11)), direction)


def test_measured_angles_are_not_the_requested_grid():
    """The regression that justifies spec 9.5.

    The fixture was produced by requesting 0, 120, 240, 360 degrees.  xtb
    returned 180, 120, -120, 0.  Any code deriving phi from the request would
    mislabel the anti minimum as the syn barrier.
    """
    frames_nm, _ = parse_scan_log(FIXTURES / "xtbscan_forward_4pt.log")
    phi = measure_angles(frames_nm, (1, 5, 8, 11))
    assert phi == pytest.approx([180.0, 120.0, -119.99, -0.01], abs=0.05)
    requested = np.array([0.0, 120.0, 240.0, 360.0])
    assert not np.allclose(phi, requested, atol=1.0)


def test_measure_angles_converts_from_one_based_indices():
    """quad is 1-based, coords are 0-based; off-by-one here is silent and fatal."""
    frames_nm, _ = parse_scan_log(FIXTURES / "xtbscan_forward_4pt.log")
    from ffrefine.geometry import dihedral_deg_frames

    assert measure_angles(frames_nm, (1, 5, 8, 11)) == pytest.approx(
        dihedral_deg_frames(frames_nm, (0, 4, 7, 10))
    )


def test_measure_angles_wraps_to_the_project_convention():
    frames_nm, _ = parse_scan_log(FIXTURES / "xtbscan_forward_4pt.log")
    phi = measure_angles(frames_nm, (1, 5, 8, 11))
    assert np.all(phi > -180.0) and np.all(phi <= 180.0)


def test_scan_leg_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ScanLeg(
            frames_nm=np.zeros((3, 4, 3)),
            energy_kj=np.zeros(2),
            phi_deg=np.zeros(3),
            direction="forward",
        )


def test_merge_pairs_by_angle_not_by_index():
    """The regression this whole design decision exists for.

    Index pairing would pair forward 180 deg with reverse 0 deg and take the
    lower energy, writing the anti minimum onto the syn point and flattening a
    20 kJ/mol barrier to nothing.
    """
    fwd = _leg(FIXTURES / "xtbscan_forward_4pt.log", "forward")
    rev = _leg(FIXTURES / "xtbscan_reverse_4pt.log", "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=7.5)

    # Output keeps the forward grid's angles.
    assert merged.phi_deg == pytest.approx(fwd.phi_deg, abs=1e-9)
    # The barrier survives: syn is still ~20.7 kJ/mol above anti.
    relative = merged.energy_kj - merged.energy_kj.min()
    assert relative[0] == pytest.approx(0.0, abs=1e-9)
    assert relative[3] == pytest.approx(20.69, abs=0.1)
    # Forward's 180 deg point has no reverse partner within tolerance (its
    # nearest reverse angle is 60 deg away): under correct angle pairing it
    # is reported unpaired. Index pairing would instead pair every forward
    # point with something and this list would come back empty.
    assert merged.unpaired_forward_deg == pytest.approx([180.0])


def test_merge_does_not_erase_a_barrier_by_pairing_on_index():
    """Nearest-angle and same-index pairing diverge starkly on this profile.

    The reverse leg traverses backwards (as a real reverse scan does), so its
    phi_deg is the forward leg's reversed. Under correct angle pairing,
    forward's 0 deg point pairs with reverse's 0 deg entry (energy 90), so the
    merged energy at 0 deg is 90. Under index pairing, forward's 0 deg (index
    0) would instead pair with reverse's FIRST entry -- 180 deg, energy 0 --
    and "take the lower energy" would write 0 onto the 0 deg point, erasing
    the 100 kJ/mol barrier between the 0 deg and 180 deg ends of the profile.
    """
    frames = np.zeros((3, 4, 3))
    fwd = ScanLeg(frames, np.array([100.0, 50.0, 0.0]), np.array([0.0, 90.0, 180.0]), "forward")
    rev = ScanLeg(frames, np.array([0.0, 50.0, 90.0]), np.array([180.0, 90.0, 0.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.energy_kj == pytest.approx([90.0, 50.0, 0.0])


def test_merge_takes_the_lower_energy_at_each_paired_point():
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 90.0]), "forward")
    rev = ScanLeg(frames, np.array([5.0, 25.0]), np.array([0.2, 90.1]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.energy_kj == pytest.approx([5.0, 20.0])
    assert merged.source == ["reverse", "forward"]


def test_merge_reports_hysteresis_and_its_worst_angle():
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 90.0]), "forward")
    rev = ScanLeg(frames, np.array([10.5, 27.0]), np.array([0.0, 90.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.max_hysteresis_kj == pytest.approx(7.0)
    assert merged.worst_angle_deg == pytest.approx(90.0)


def test_merge_pairs_across_the_wrap_boundary():
    """-179 and +179 are two degrees apart, not 358."""
    frames = np.zeros((1, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0]), np.array([179.0]), "forward")
    rev = ScanLeg(frames, np.array([4.0]), np.array([-179.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.energy_kj == pytest.approx([4.0])
    assert merged.unpaired_forward_deg == []


def test_merge_reports_unpaired_points_rather_than_dropping_them():
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 90.0]), "forward")
    rev = ScanLeg(frames, np.array([5.0, 1.0]), np.array([0.0, 45.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    # The forward point at 90 has no partner, so it keeps its own energy.
    assert merged.energy_kj == pytest.approx([5.0, 20.0])
    assert merged.unpaired_forward_deg == pytest.approx([90.0])
    assert merged.unpaired_reverse_deg == pytest.approx([45.0])


def test_merge_refuses_when_two_forward_points_claim_one_reverse_point():
    """A degenerate grid; guessing which pairing was meant would be a coin flip."""
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 1.0]), "forward")
    rev = ScanLeg(np.zeros((1, 4, 3)), np.array([5.0]), np.array([0.5]), "reverse")

    with pytest.raises(ValidationError, match="both|ambiguous"):
        merge_bidirectional(fwd, rev, tol_deg=5.0)


def test_merge_keeps_the_frame_belonging_to_the_chosen_energy():
    """Taking the reverse energy but the forward geometry would be incoherent."""
    fwd_frames = np.zeros((1, 4, 3))
    rev_frames = np.ones((1, 4, 3))
    fwd = ScanLeg(fwd_frames, np.array([10.0]), np.array([0.0]), "forward")
    rev = ScanLeg(rev_frames, np.array([5.0]), np.array([0.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.frames_nm[0] == pytest.approx(rev_frames[0])


def _sweep(n: int = 12, natoms: int = 8, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """Frames sweeping a full turn: half the atoms rotate about z.

    Two properties matter and both are easy to get wrong.  The motion is
    **non-rigid** (only half the atoms move), because Kabsch alignment removes
    rigid motion entirely -- a whole-molecule translation would produce an RMSD
    of zero no matter how large it is.  And it is **periodic**, so the last
    frame sits one step from the first, which is what a real torsion scan looks
    like and what makes the cyclic wrap check meaningful.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(natoms, 3)) * 0.1
    half = natoms // 2
    out = []
    for step in range(n):
        angle = 2.0 * np.pi * step / n
        frame = base.copy()
        c, s = np.cos(angle), np.sin(angle)
        x, y = base[half:, 0].copy(), base[half:, 1].copy()
        frame[half:, 0] = c * x - s * y
        frame[half:, 1] = s * x + c * y
        if jitter:
            frame = frame + rng.normal(size=(natoms, 3)) * jitter
        out.append(frame)
    return np.stack(out)


def _phi(frames: np.ndarray) -> np.ndarray:
    return np.linspace(-180.0, 180.0, len(frames), endpoint=False)


def test_find_hops_is_silent_on_a_smooth_scan():
    """A clean sweep has uniform steps and no hops."""
    frames = _sweep()
    assert find_hops(frames, list(range(frames.shape[1])), _phi(frames)) == []


def test_find_hops_is_silent_on_a_jittered_scan():
    """Real scans wobble; wobble is not a hop.  Exercises the MAD branch."""
    frames = _sweep(jitter=0.002)
    assert find_hops(frames, list(range(frames.shape[1])), _phi(frames)) == []


def test_find_hops_flags_a_planted_spike():
    """Half the atoms jump between frames 6 and 7 -- a non-rigid rearrangement.

    Also the regression for the uniform-step case: a clean sweep has identical
    adjacent RMSDs, so the MAD is exactly zero.  An implementation that bails
    out when MAD is zero would find nothing here, which is backwards -- a
    perfectly uniform run of steps is the *easiest* baseline to spot a spike
    against, not an impossible one.
    """
    frames = _sweep()
    half = frames.shape[1] // 2
    frames[7:, :half] += 0.5

    hops = find_hops(frames, list(range(frames.shape[1])), _phi(frames), cyclic=False)

    assert [h.index for h in hops] == [7]
    assert hops[0].rmsd_nm > hops[0].threshold_nm


def test_find_hops_ignores_hydrogens():
    """Spec 4.3 says heavy-atom RMSD; a spinning methyl is not a hop."""
    frames = _sweep(natoms=8)
    frames[5:, 4:] += 2.0  # only the atoms we will exclude move
    heavy = [0, 1, 2, 3]

    assert find_hops(frames, heavy, _phi(frames), cyclic=False) == []


def test_find_hops_checks_the_wrap_pair_only_when_cyclic():
    """A full-turn scan's last and first frames are one step apart in angle."""
    frames = _sweep()
    half = frames.shape[1] // 2
    frames[-1, :half] += 0.5

    cyclic = {
        h.index for h in find_hops(frames, list(range(frames.shape[1])), _phi(frames), cyclic=True)
    }
    linear = {
        h.index for h in find_hops(frames, list(range(frames.shape[1])), _phi(frames), cyclic=False)
    }

    assert 0 in cyclic  # the wrap pair (last frame -> frame 0)
    assert 0 not in linear  # ... and only when asked for


def test_find_hops_survives_identical_frames():
    """No motion at all: median and MAD are both zero, and nothing is a hop."""
    frames = np.zeros((6, 4, 3))
    assert find_hops(frames, [0, 1, 2, 3], _phi(frames), cyclic=False) == []


def test_find_hops_needs_enough_points_to_judge_an_outlier():
    """Two frames give one gap; calling it an outlier against itself is nonsense."""
    frames = np.zeros((2, 4, 3))
    frames[1, :2] += 5.0
    assert find_hops(frames, [0, 1, 2, 3], np.array([0.0, 90.0]), cyclic=False) == []
