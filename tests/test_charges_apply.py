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
    """Silently leaving the total wrong would be the worst option.

    Every atom is selected and sits in its own charge group, so each group's
    nonzero residual has nowhere to go. (This used target_total=1, which is
    now refused earlier as a whole-electron offset -- ruling R21 -- so it no
    longer reached this path; the residual alone is enough to reach it.)
    """
    top = read_topology(butane_itp)
    for atom in top.atoms:
        atom.cgnr = atom.nr

    with pytest.raises(ValidationError, match="nothing left to redistribute"):
        apply_charges(top, _fitted(), selected=list(range(1, 15)), target_total=0)


def test_the_total_is_exact_after_rounding(butane_itp):
    """Rounding 14 charges to 4 decimals leaves a residual; it must be absorbed."""
    rng = np.random.default_rng(0)
    top = read_topology(butane_itp)
    q = rng.normal(size=14) * 0.2
    q -= q.sum() / 14.0  # sums to zero, but not to any multiple of 1e-4

    out = apply_charges(top, q, selected=None, target_total=0)

    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)
    assert all(a.after == pytest.approx(round(a.after, 4)) for a in out)


def test_a_whole_electron_offset_is_refused_in_selected_mode(butane_itp):
    """Final review I2 (ruling R21, amending R10/R11).

    Butane's charges sum to 0, so a target of -1 leaves a whole electron that
    selected mode could only spread over the three hydrogens sharing C1's
    charge group (taking them from +0.090 to about -0.2). That is not a
    rounding correction; it is refused, naming both remedies.
    """
    top = read_topology(butane_itp)

    with pytest.raises(ValidationError, match="--charge-mode all") as exc:
        apply_charges(top, _fitted(), selected=[1], target_total=-1)
    assert "correct the topology" in str(exc.value)


def test_a_small_offset_is_still_spread_over_the_recipients(butane_itp):
    """Offsets up to CHARGE_TOLERANCE (0.05 e) are rounding-sized and still absorbed.

    C1's input charge is nudged to -0.290 so the topology sums to -0.02; with
    C1 selected, the group residual and the +0.02 offset both land on C1's
    group-mates, equally, and nothing outside that group moves.
    """
    top = read_topology(butane_itp)
    top.atoms[0].charge = -0.290

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)
    assert out[0].after == pytest.approx(-0.35)
    assert [a.nr for a in out if a.source == "redistributed"] == [2, 3, 4]
    deltas = [out[i].after - out[i].before for i in (1, 2, 3)]
    assert deltas == pytest.approx([(0.06 + 0.02) / 3] * 3, abs=1e-4)
    assert all(a.after == a.before for a in out[4:])


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


def test_a_whole_electron_offset_on_a_zero_residual_group_is_refused(butane_itp):
    """The R12 structure (atom 6 selected, its own residual zero) with a -1 offset.

    This used to spread the whole electron over atoms 5 and 7. It is now
    refused: nothing about a zero-residual selection makes a whole-electron
    disagreement between --charge and the topology a rounding matter.
    """
    top = read_topology(butane_itp)
    fitted = _fitted()
    fitted[5] = 0.090  # atom 6 (index 5): matches its topology charge exactly

    with pytest.raises(ValidationError, match="--charge-mode all"):
        apply_charges(top, fitted, selected=[6], target_total=-1)


def test_a_small_global_offset_lands_on_the_selected_atoms_own_group(butane_itp):
    """A small global offset routes to group recipients, not the largest atom.

    Atom 6 (H21, cgnr 2 = atoms 5-7) is selected with a fitted value equal to
    its topology charge, so its group's own residual is zero -- the fallback
    bug (round 2, commit 8fd5400) skipped collecting that group's recipients in
    this case and instead dumped the global offset on argmax(|after|), an atom
    outside cgnr 2 entirely. C1's input charge is nudged to -0.290 so the
    topology sums to -0.02 -- a +0.02 offset, within CHARGE_TOLERANCE -- and C1
    is then the unique largest-magnitude atom, exactly where that bug would put
    it. The offset must go to atom 6's own group-mates, atoms 5 and 7.
    """
    top = read_topology(butane_itp)
    top.atoms[0].charge = -0.290
    fitted = _fitted()
    fitted[5] = 0.090  # atom 6 (index 5): matches its topology charge exactly

    out = apply_charges(top, fitted, selected=[6], target_total=0)

    # Atom 6 keeps its fitted value exactly.
    assert out[5].after == pytest.approx(0.090)
    assert out[5].source == "fitted"

    # The +0.02 offset lands only on atom 6's group-mates (atoms 5 and 7 of
    # cgnr 2), spread evenly between them, never on C1 (atom 1) or C4 (atom 11).
    redistributed = sorted(a.nr for a in out if a.source == "redistributed")
    assert redistributed == [5, 7]
    deltas = {a.nr: a.after - a.before for a in out if a.nr in (5, 7)}
    assert deltas[5] == pytest.approx(0.01, abs=1e-9)
    assert deltas[7] == pytest.approx(0.01, abs=1e-9)
    assert out[0].after == out[0].before == pytest.approx(-0.290)

    # Total must be exact.
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)


def test_no_recipients_with_nonzero_offset_is_refused(butane_itp):
    """Nonzero global offset with no recipients anywhere must be refused.

    If every atom is selected and there are no recipients to absorb a global
    offset (e.g. from an explicit --charge), the function must refuse rather
    than silently dumping it on one atom. The offset is a small +0.02 e (C1
    nudged to -0.290): it used to be a whole electron, which ruling R21 now
    refuses earlier, before this path is reached.
    """
    top = read_topology(butane_itp)
    for atom in top.atoms:  # every atom in its own charge group
        atom.cgnr = atom.nr
    top.atoms[0].charge = -0.290  # topology sums to -0.02

    # All atoms selected, fitted equals topology (no group residuals),
    # but target differs by 0.02 (global offset nonzero)
    fitted = np.array([a.charge for a in top.atoms], dtype=float)

    with pytest.raises(ValidationError, match="nothing left.*global offset"):
        apply_charges(top, fitted, selected=list(range(1, 15)), target_total=0)


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


# Six-decimal charges, as GAFF/antechamber RESP output carries; each charge
# group still sums to zero, so the molecule stays neutral.
SIX_DECIMAL_BUTANE = {
    "C1": -0.270123,
    "H11": 0.090041,
    "H12": 0.090041,
    "H13": 0.090041,
    "C2": -0.180246,
    "H21": 0.090123,
    "H22": 0.090123,
    "C3": -0.180246,
    "H31": 0.090123,
    "H32": 0.090123,
    "C4": -0.270123,
    "H41": 0.090041,
    "H42": 0.090041,
    "H43": 0.090041,
}


def _six_decimal_butane(butane_itp):
    top = read_topology(butane_itp)
    for atom in top.atoms:
        atom.charge = SIX_DECIMAL_BUTANE[atom.name]
    return top


def test_output_precision_follows_the_input_never_below_four(butane_itp):
    """Final review I3 (ruling R22): decimals = max(4, the input's own precision)."""
    from ffrefine.charges.apply import charge_decimals

    assert charge_decimals(read_topology(butane_itp)) == 4  # butane is written to 3
    assert charge_decimals(_six_decimal_butane(butane_itp)) == 6

    top = read_topology(butane_itp)
    top.atoms[1].charge = 0.12345678  # one atom at 8 decimals sets the precision
    assert charge_decimals(top) == 8


def test_untouched_atoms_keep_their_exact_six_decimal_input_values(butane_itp):
    """Selecting one atom must not rewrite a dozen others by rounding them.

    Rounding every atom to 4 decimals rewrote every atom of a 6-decimal
    topology, labelled the untouched ones "redistributed" and logged them as
    absorbing the redistribution residual (ruling R22).
    """
    top = _six_decimal_butane(butane_itp)
    fitted = _fitted()
    fitted[0] = -0.3512345678

    out = apply_charges(top, fitted, selected=[1], target_total=0)

    for a in out[4:]:  # everything outside C1's charge group
        assert a.after == a.before == SIX_DECIMAL_BUTANE[a.name]
        assert a.source == "unchanged"
    assert [a.nr for a in out if a.source == "redistributed"] == [2, 3, 4]
    assert out[0].source == "fitted"
    assert out[0].after == pytest.approx(-0.351235, abs=1e-12)  # rounded at 6, not 4
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-9)
    assert all(a.after == pytest.approx(round(a.after, 6), abs=1e-12) for a in out)


def test_an_explicit_precision_coarser_than_the_input_is_refused(butane_itp):
    """Untouched 6-decimal atoms cannot sum exactly with 4-decimal changed ones."""
    top = _six_decimal_butane(butane_itp)

    with pytest.raises(ValidationError, match="decimal"):
        apply_charges(top, _fitted(), selected=[1], target_total=0, decimals=4)
