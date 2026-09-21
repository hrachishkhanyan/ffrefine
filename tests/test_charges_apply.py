import numpy as np
import pytest

from ffrefine.charges.apply import apply_charges
from ffrefine.errors import ValidationError
from ffrefine.io.topology import read_topology


def _fitted(natoms: int = 14) -> np.ndarray:
    """A fitted set that differs from butane's and still sums to zero."""
    q = np.full(natoms, 0.05)
    q[[0, 10]] = -0.35
    return q


def test_all_mode_replaces_every_charge_and_keeps_the_total_exact(butane_itp):
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=None, target_total=0)

    assert {a.source for a in out} == {"fitted"}
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)


def test_selected_mode_changes_only_the_named_atoms(butane_itp):
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    assert out[0].source == "fitted"
    assert out[0].after == pytest.approx(-0.35)
    # Everything outside charge group 1 is untouched.
    assert all(a.source == "unchanged" for a in out[4:])


def test_the_residual_lands_in_the_selected_atoms_charge_group(butane_itp):
    """Spec 5.3: spread over the atoms in the same charge group."""
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    recipients = [a.nr for a in out if a.source == "redistributed"]
    assert recipients == [2, 3, 4]  # the rest of cgnr 1
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)


def test_the_residual_falls_back_to_bonded_heavy_atoms(butane_itp):
    """A charge group with nothing else in it must not dead-end."""
    top = read_topology(butane_itp)
    for atom in top.atoms:  # every atom in its own charge group
        atom.cgnr = atom.nr

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    recipients = [a.nr for a in out if a.source == "redistributed"]
    assert recipients == [5]  # C2, the only heavy atom bonded to C1
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)


def test_no_recipient_anywhere_is_refused(butane_itp):
    """Silently leaving the total wrong would be the worst option."""
    top = read_topology(butane_itp)
    for atom in top.atoms:
        atom.cgnr = atom.nr

    with pytest.raises(ValidationError, match="redistribute"):
        apply_charges(top, _fitted(), selected=list(range(1, 15)), target_total=1)


def test_the_total_is_exact_after_rounding(butane_itp):
    """Rounding 14 charges to 4 decimals leaves a residual; it must be absorbed."""
    rng = np.random.default_rng(0)
    top = read_topology(butane_itp)
    q = rng.normal(size=14) * 0.2
    q -= q.sum() / 14.0  # sums to zero, but not to any multiple of 1e-4

    out = apply_charges(top, q, selected=None, target_total=0)

    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)
    assert all(a.after == pytest.approx(round(a.after, 4)) for a in out)


def test_a_nonzero_target_total_is_honoured(butane_itp):
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=-1)

    assert sum(a.after for a in out) == pytest.approx(-1.0, abs=1e-6)


def test_an_out_of_range_atom_is_refused(butane_itp):
    top = read_topology(butane_itp)

    with pytest.raises(ValidationError, match="15"):
        apply_charges(top, _fitted(), selected=[15], target_total=0)


def test_the_before_column_is_the_topology_not_the_fit(butane_itp):
    """charges.json reports what changed, so 'before' must be the input value."""
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    assert out[0].before == pytest.approx(-0.270)
    assert out[0].name == "C1"


def test_redistribution_is_per_charge_group_not_global(butane_itp):
    """Spec 5.3: residual is spread within each charge group, not between them.

    Butane's charge groups are: 1-4 (cgnr 1), 5-7 (cgnr 2), 8-10 (cgnr 3),
    11-14 (cgnr 4). Selecting atoms in two different groups must not allow
    charge to flow between them.
    """
    top = read_topology(butane_itp)

    # Give atoms 1 and 11 different fitted values to test per-group redistribution.
    # With pooled redistribution, charge would leak between groups; with per-group,
    # each group absorbs its own residual.
    q = np.full(14, 0.05)
    q[0] = -0.40  # atom 1: larger negative change
    q[10] = -0.30  # atom 11: smaller negative change

    out = apply_charges(top, q, selected=[1, 11], target_total=0)

    # Group 1: atoms 1-4
    group1_after = sum(a.after for a in out if 1 <= a.nr <= 4)
    # Group 4: atoms 11-14
    group4_after = sum(a.after for a in out if 11 <= a.nr <= 14)

    # Each group's sum should be preserved to within rounding precision
    assert group1_after == pytest.approx(0.0, abs=1e-4)
    assert group4_after == pytest.approx(0.0, abs=1e-4)
    # Molecular total must be exact
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)
