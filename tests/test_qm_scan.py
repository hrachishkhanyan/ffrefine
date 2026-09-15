import numpy as np
import pytest
from pathlib import Path

from ffrefine.errors import ValidationError
from ffrefine.geometry import kabsch_rmsd
from ffrefine.qm.scan import (
    UNIFORM_STEP_FACTOR,
    Hop,
    ScanLeg,
    find_hops,
    measure_angles,
    merge_bidirectional,
)
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

    This is the ordinary scaled-MAD path, not the uniform-step one.  A clean
    sweep does *not* have identical adjacent RMSDs: Kabsch realigns every pair
    independently, so the ordinary steps differ in the fourth decimal and leave
    a healthy MAD around 4e-3 nm.  The uniform-step fallback needs a baseline
    that returns to the same geometries, and is covered by
    ``test_find_hops_flags_a_spike_against_a_perfectly_uniform_baseline``.
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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("noise_nm", [0.0, 1e-18, 1e-16, 1e-14, 1e-12])
def test_find_hops_is_silent_on_a_molecule_that_never_moved(seed, noise_nm):
    """A scan that never left its input geometry has no hops, only rounding noise.

    Kabsch superposition of two *identical* real geometries does not return
    exactly 0.0 -- it returns the SVD's rounding residue, around 1e-17 nm.  So
    a detector that recognises "nothing moved" by testing the RMSD against
    exact zero never recognises it at all: it runs its outlier test over pure
    floating-point noise instead, where some pair is always a few ulp above the
    rest.  That reports a conformational hop across a displacement of 1e-17 nm,
    ten orders of magnitude below a chemical bond.

    The noise levels here are all far below any physical motion, so every one
    of them means the same thing -- the molecule never moved -- and none of
    them may produce a hop.  The parametrisation is the point: whether a
    spurious hop appears is a matter of which pair happens to round up, so a
    single seed would pass or fail by luck.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(8, 3)) * 0.1
    frames = np.stack([base + rng.normal(size=(8, 3)) * noise_nm for _ in range(12)])

    assert find_hops(frames, list(range(8)), _phi(frames)) == []


def _alternating(noise_nm: float = 0.0, seed: int = 0) -> np.ndarray:
    """Frames alternating between two fixed conformers: every step is the same size.

    A subset rotation will not do here.  Kabsch realigns each pair
    independently, so even a constant angle increment yields adjacent RMSDs
    that differ in the fourth decimal -- a perfectly healthy MAD.  Returning to
    the *same* two geometries is what makes every adjacent RMSD identical, and
    identical only to within rounding: the MAD collapses to about 1e-17 nm
    rather than to zero, which is the case the threshold has to survive.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(8, 3)) * 0.1
    other = base.copy()
    other[4:, 0] += 0.3
    frames = [(base if i % 2 == 0 else other) for i in range(12)]
    if noise_nm:
        frames = [f + rng.normal(size=(8, 3)) * noise_nm for f in frames]
    return np.stack(frames)


def test_find_hops_treats_a_rounding_level_mad_as_a_uniform_baseline():
    """A MAD made of rounding noise sets no scale, so it must not set the threshold.

    Steps identical to within rounding leave a MAD around 1e-16 nm, which is
    not the exact zero the uniform-step rule used to wait for.  Routing on
    ``mad > 0`` therefore sent this case down the scaled-MAD path, where the
    threshold sits a few ulp above the median and only the width of that
    particular rounding cluster decides whether ordinary steps survive it.
    Nothing about the geometry does.  Treating a negligible MAD as the zero it
    represents puts the decision back on the step sizes.
    """
    frames = _alternating(noise_nm=1e-15)

    assert find_hops(frames, list(range(8)), _phi(frames)) == []


def test_find_hops_flags_a_spike_against_a_perfectly_uniform_baseline():
    """Ten identical steps then one nine times larger: the one is the hop.

    This is the uniform-step branch's detection case, which no other test
    covers -- and which nothing reached at all while the branch waited for a
    MAD of exactly zero, because the ten "identical" steps still differ by an
    ulp and leave a MAD around 3e-17 nm.  Keep this test honest by checking it
    still routes through the fallback: if the ordinary scaled-MAD path can
    satisfy it, it has stopped testing the branch it names.
    """
    frames = _alternating()
    frames[-1, 4:] += 1.5

    hops = find_hops(frames, list(range(8)), _phi(frames), cyclic=False)

    assert [h.index for h in hops] == [11]
    assert hops[0].rmsd_nm > hops[0].threshold_nm
    # The reported threshold says which rule produced it: the uniform-step
    # multiple of the ordinary step, not the median a hair's breadth above it.
    ordinary_step = kabsch_rmsd(frames[0], frames[1])
    assert hops[0].threshold_nm == pytest.approx(UNIFORM_STEP_FACTOR * ordinary_step)
