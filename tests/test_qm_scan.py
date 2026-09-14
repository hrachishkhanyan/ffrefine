import numpy as np
import pytest
from pathlib import Path

from ffrefine.qm.scan import ScanLeg, measure_angles
from ffrefine.qm.xtb import parse_scan_log

FIXTURES = Path(__file__).parent.parent / "fixtures" / "butane" / "xtb"


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
