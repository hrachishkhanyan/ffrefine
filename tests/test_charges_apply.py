import numpy as np
import pytest

from ffrefine.charges.apply import apply_charges
from ffrefine.errors import ValidationError
from ffrefine.io.topology import read_topology


def _fitted(natoms: int = 14) -> np.ndarray:
    """A fitted set that differs from butane's and sums to zero.

    Atoms 1 and 11 (indices 0 and 10) are pinned at -0.35 e so the tests that
    only touch those two atoms keep their existing expected values; the other
    twelve atoms split +0.70 e evenly so the set actually sums to zero. (An
    earlier version of this helper put 0.05 on those twelve atoms, which sums
    to -0.1 e, not zero -- the docstring claimed otherwise.)
    """
    q = np.full(natoms, 0.70 / 12)
    q[[0, 10]] = -0.35
    assert q.sum() == pytest.approx(0.0, abs=1e-9)
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


def test_global_offset_lands_on_group_recipients_not_drifted_atoms(butane_itp):
    """Global offset (explicit --charge) routes to group recipients, not largest atom.

    Atom 6 (H21, cgnr 2 = atoms 5-7) is selected with a fitted value equal to
    its topology charge, so its group's own residual is zero -- the fallback
    bug (round 2, commit 8fd5400) skips collecting that group's recipients in
    this case and instead dumps the global offset on argmax(|after|), which
    for butane is atom 1 or atom 11 (both -0.270, C1 winning the tie) -- an
    atom outside cgnr 2 entirely. The fix must route the offset to atom 6's
    own group-mates (atoms 5 and 7), not to C1.
    """
    top = read_topology(butane_itp)

    fitted = _fitted()
    fitted[5] = 0.090  # atom 6 (index 5): matches its topology charge exactly

    out = apply_charges(top, fitted, selected=[6], target_total=-1)

    # Atom 6 keeps its fitted value exactly.
    assert out[5].after == pytest.approx(0.090)
    assert out[5].source == "fitted"

    # The -1 offset lands only on atom 6's group-mates (atoms 5 and 7 of cgnr 2),
    # spread evenly between them, never on C1 (atom 1) or C4 (atom 11).
    redistributed = sorted(a.nr for a in out if a.source == "redistributed")
    assert redistributed == [5, 7]
    deltas = {a.nr: a.after - a.before for a in out if a.nr in (5, 7)}
    assert deltas[5] == pytest.approx(deltas[7])

    # Total must be exact.
    assert sum(a.after for a in out) == pytest.approx(-1.0, abs=1e-6)


def test_no_recipients_with_nonzero_offset_is_refused(butane_itp):
    """Nonzero global offset with no recipients anywhere must be refused.

    If every atom is selected and there are no recipients to absorb a global
    offset (e.g. from an explicit --charge), the function must refuse rather
    than silently dumping it on one atom.
    """
    top = read_topology(butane_itp)
    for atom in top.atoms:  # every atom in its own charge group
        atom.cgnr = atom.nr

    # All atoms selected, fitted equals topology (no group residuals),
    # but target differs by 1 (global offset nonzero)
    fitted = np.array([a.charge for a in top.atoms], dtype=float)

    with pytest.raises(ValidationError, match="nothing left"):
        apply_charges(top, fitted, selected=list(range(1, 15)), target_total=1)


def test_all_mode_refuses_fitted_charges_that_do_not_sum_to_target(butane_itp):
    """A large drift in ``all`` mode must be refused, not silently absorbed.

    The ESP job runs at charge=target_total, so legitimately fitted charges
    already sum to the target; only rounding can drift the total, by at most
    ~natoms * 0.5 * 10**-decimals. A drift far larger than that (here 0.2 e,
    against a 14-atom rounding budget of ~0.0007 e) means something upstream
    produced fitted charges that do not sum to the target, and must raise
    rather than be dumped onto argmax(|after|).
    """
    top = read_topology(butane_itp)
    q = _fitted()
    q[0] += 0.2  # now sums to 0.2 e away from target_total=0 -- not rounding-sized

    with pytest.raises(ValidationError, match="drift"):
        apply_charges(top, q, selected=None, target_total=0)
