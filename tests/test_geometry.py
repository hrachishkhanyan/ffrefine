import numpy as np
import pytest
from ffrefine import geometry as g


def test_wrap_maps_minus_180_to_positive_180():
    assert g.wrap_deg(180.0) == pytest.approx(180.0)
    assert g.wrap_deg(-180.0) == pytest.approx(180.0)


def test_wrap_range():
    a = np.array([-540.0, -181.0, -179.0, 0.0, 181.0, 359.0, 361.0])
    w = g.wrap_deg(a)
    assert np.all(w > -180.0) and np.all(w <= 180.0)
    assert w[2] == pytest.approx(-179.0)
    assert w[5] == pytest.approx(-1.0)
    assert w[6] == pytest.approx(1.0)


def _four_points(phi_deg):
    """Four points whose dihedral is exactly phi_deg, by construction."""
    phi = np.deg2rad(phi_deg)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.5, 0.0, 0.0])
    p0 = np.array([-0.5, 1.0, 0.0])
    p3 = np.array([2.0, np.cos(phi), np.sin(phi)])
    return p0, p1, p2, p3


@pytest.mark.parametrize("phi", [0.0, 60.0, 90.0, 120.0, 180.0, -60.0, -120.0])
def test_dihedral_recovers_constructed_angle(phi):
    p0, p1, p2, p3 = _four_points(phi)
    assert g.dihedral_deg(p0, p1, p2, p3) == pytest.approx(g.wrap_deg(phi), abs=1e-8)


def test_dihedral_sign_flips_under_mirror():
    p0, p1, p2, p3 = _four_points(60.0)
    m = np.diag([1.0, 1.0, -1.0])
    assert g.dihedral_deg(p0 @ m, p1 @ m, p2 @ m, p3 @ m) == pytest.approx(-60.0, abs=1e-8)


def test_dihedral_frames_shape_and_values():
    frames = np.stack([np.stack(_four_points(p)) for p in (0.0, 60.0, -60.0)])
    out = g.dihedral_deg_frames(frames, (0, 1, 2, 3))
    assert out.shape == (3,)
    assert out == pytest.approx([0.0, 60.0, -60.0], abs=1e-8)


def test_kabsch_zero_for_rotated_translated_copy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 3))
    t = 0.7
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    b = a @ rot.T + np.array([3.0, -1.0, 2.0])
    assert g.kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_detects_real_difference():
    a = np.zeros((4, 3))
    a[1] = [1, 0, 0]
    a[2] = [0, 1, 0]
    a[3] = [0, 0, 1]
    b = a.copy()
    b[3] = [0, 0, 2.0]
    assert g.kabsch_rmsd(a, b) > 0.1
