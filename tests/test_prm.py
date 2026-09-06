import numpy as np
import pytest
from ffrefine.io.prm import write_prm_block
from ffrefine.io.topology import read_topology


def test_prm_uses_atom_types_not_indices(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([4.184]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=True,
    )
    assert "CG331" in text and "CG321" in text


def test_prm_is_in_kcal(tmp_path, butane_itp):
    """4.184 kJ/mol must be written as 1.0 kcal/mol."""
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([4.184]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=True,
    )
    numbers = [
        ln.split()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("!", "DIHEDRALS", "END"))
    ]
    assert float(numbers[0][4]) == pytest.approx(1.0, abs=1e-6)


def test_prm_has_dihedrals_header(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([1.0]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=True,
    )
    assert text.strip().startswith("DIHEDRALS") or "DIHEDRALS" in text.splitlines()[0]


def test_prm_warns_loudly_without_distribute(tmp_path, butane_itp):
    """Spec section 4.8: a type-level block is meaningless without --distribute."""
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([1.0]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=False,
    )
    assert "WARNING" in text.upper()
    assert "distribute" in text.lower()


def test_prm_no_warning_with_distribute(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([1.0]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=True,
    )
    assert "WARNING" not in text.upper()


def test_prm_one_line_per_multiplicity(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([1.0, 0.5]),
        np.array([0.0, 180.0]),
        np.array([3, 2]),
        tmp_path / "o.prm",
        distribute=True,
    )
    data = [
        ln
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("!", "DIHEDRALS", "END"))
    ]
    assert len(data) == 2
