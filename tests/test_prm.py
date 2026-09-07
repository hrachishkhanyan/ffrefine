import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.prm import write_prm_block
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import write_refit_topology
from ffrefine.mm.torsionset import (
    collect_torsion_set,
    distinct_entry_keys,
    distribution_count,
)
from ffrefine.units import kj_to_kcal


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
    """4.184 kJ/mol must be written as 1.0 kcal/mol.

    Uses ``distribute=False`` so this asserts the unit conversion *alone*.
    Under ``--distribute`` the same fitted k is additionally divided by N_T
    (see ``test_prm_kchi_matches_itp_force_constant_under_distribute``), which
    would confound a pure units assertion.
    """
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([4.184]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=False,
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
    # CHARMM allows leading '!' comments, and both branches emit some, so the
    # assertion is that DIHEDRALS is the first *non-comment* line.
    body = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("!")]
    assert body[0].strip() == "DIHEDRALS"


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


def test_prm_rejects_out_of_range_quad(tmp_path, butane_itp):
    """Fix round 1, finding 1: a raw list index must not silently wrap or
    raise a bare IndexError -- it must raise ValidationError."""
    t = read_topology(butane_itp)
    with pytest.raises(ValidationError):
        # index 0 would wrap to atoms[-1] under naive 1-based -> 0-based
        # indexing, silently picking the LAST atom instead of erroring.
        write_prm_block(
            t,
            (0, 5, 8, 11),
            np.array([1.0]),
            np.array([0.0]),
            np.array([3]),
            tmp_path / "o.prm",
            distribute=True,
        )
    with pytest.raises(ValidationError):
        write_prm_block(
            t,
            (999, 5, 8, 11),
            np.array([1.0]),
            np.array([0.0]),
            np.array([3]),
            tmp_path / "o.prm",
            distribute=True,
        )


def test_prm_six_char_atom_type_does_not_fuse_columns(tmp_path, butane_itp):
    """Fix round 1, finding 2: a 6-char CGenFF type (e.g. CG2R61) must not
    fuse with the adjacent column when padding alone provides no gap."""
    t = read_topology(butane_itp)
    t.atoms[0].atype = "CG2R61"
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([1.0]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=True,
    )
    data = [
        ln
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("!", "DIHEDRALS", "END"))
    ]
    fields = data[0].split()
    # Four separately-parseable type tokens, not one fused blob -- checked by
    # exact equality, not a substring, so a fused "CG2R61CG321" would fail.
    assert fields[:4] == ["CG2R61", "CG321", "CG321", "CG331"]


def _data_lines(text: str) -> list[str]:
    return [
        ln
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("!", "DIHEDRALS", "END"))
    ]


def test_prm_kchi_matches_itp_force_constant_under_distribute(tmp_path, butane_itp):
    """Final review, P0: the two artifacts of one --distribute fit must agree.

    ``write_refit_topology`` divides the fitted k by N_T (9 distinct
    quadruplets on butane); ``write_prm_block`` used to write it undivided, so
    the CHARMM Kchi was 9x the .itp force constant for the same fit.  A CHARMM
    Kchi is a per-quadruplet constant summed over every matching quadruplet,
    exactly as GROMACS sums its [ dihedrals ] lines, so the two must be equal.
    """
    t = read_topology(butane_itp)
    quad = (1, 5, 8, 11)
    k = np.array([5.0])
    delta = np.array([0.0])
    mult = np.array([3])

    write_refit_topology(t, quad, k, delta, mult, tmp_path / "d.itp", distribute=True)
    text = write_prm_block(t, quad, k, delta, mult, tmp_path / "d.prm", distribute=True)

    # Pull the distributed k out of the .itp.  Matching on the writer's own
    # "distributed over" comment rather than on "nonzero k": butane carries
    # dihedrals about OTHER bonds that keep their original 0.66944 kJ/mol.
    itp_lines = [
        ln for ln in (tmp_path / "d.itp").read_text().splitlines() if "distributed over" in ln
    ]
    assert len(itp_lines) == 9, f"expected 9 distributed lines, got {len(itp_lines)}"
    itp_ks = {round(float(ln.split(";")[0].split()[6]), 9) for ln in itp_lines}
    assert len(itp_ks) == 1, f"expected one distributed force constant, got {itp_ks}"
    itp_k_kj = itp_ks.pop()

    # Tolerances are the printed field resolutions: %12.5f in the .itp,
    # %10.4f in the .prm.  Pre-fix the .prm carried 9x this (1.1950 kcal/mol
    # against 0.1328), so nothing this loose can hide the defect.
    assert itp_k_kj == pytest.approx(5.0 / 9.0, abs=1e-5)
    prm_kchi = float(_data_lines(text)[0].split()[4])
    assert prm_kchi == pytest.approx(kj_to_kcal(itp_k_kj), abs=1e-4)


def test_prm_undistributed_kchi_is_the_whole_fitted_k(tmp_path, butane_itp):
    """Without --distribute the .itp carries the full k on the target quad, so
    the .prm must too -- the 1/N_T scaling is conditional, not unconditional."""
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([5.0]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=False,
    )
    assert float(_data_lines(text)[0].split()[4]) == pytest.approx(kj_to_kcal(5.0), abs=1e-4)


def test_prm_distribute_note_names_the_quadruplets(tmp_path, butane_itp):
    """The block must say which quadruplets the entry covers, and how many."""
    t = read_topology(butane_itp)
    text = write_prm_block(
        t,
        (1, 5, 8, 11),
        np.array([5.0]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "o.prm",
        distribute=True,
    )
    assert "9 distinct" in text
    assert "1-5-8-11" in text
    assert "7-5-8-10" in text
    # Every quadruplet of T is named, so a reader can check the denominator.
    for key in distinct_entry_keys(t, (1, 5, 8, 11)):
        assert "-".join(str(i) for i in key[:4]) in text


def test_distribution_count_is_quadruplets_not_term_lines(butane_itp):
    """Rulings R14/R17: 10 term-lines, 9 distinct quadruplets.  N_T is 9."""
    t = read_topology(butane_itp)
    assert len(collect_torsion_set(t, (1, 5, 8, 11))) == 10
    assert distribution_count(t, (1, 5, 8, 11)) == 9
