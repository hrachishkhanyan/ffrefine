from pathlib import Path

import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.mm import selfcheck

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAPECITABINE = FIXTURES / "capecitabine_gromacs"


@pytest.mark.needs_gmx
def test_self_consistency_passes_on_butane(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """Spec section 4.5: agreement better than 1e-3 kJ/mol at every point."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 12)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path
    )
    assert r.passed
    assert r.max_abs_error < 1e-3


@pytest.mark.needs_gmx
def test_self_check_samples_a_real_torsion_range(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 12)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path
    )
    assert r.phi_deg.ptp() > 180.0
    assert np.abs(r.analytic_difference).max() > 0.5


@pytest.mark.needs_gmx
def test_self_check_raises_when_tolerance_impossible(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """A deliberately absurd tolerance must abort rather than warn."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 6)
    with pytest.raises(ValidationError, match="self-consistency") as excinfo:
        selfcheck.check_torsion_removal(
            Binaries(gmx=gmx_bin),
            t,
            (1, 5, 8, 11),
            s,
            frames,
            tmp_path,
            tolerance=1e-18,
        )
    msg = str(excinfo.value)
    assert "10 term(s): 10 written in the topology and 0 resolved from [ dihedraltypes ]" in msg
    assert "full_dump.stdout" in msg
    # The old claim is gone: typed terms are no longer refused before the run.
    assert "checked before the run" not in msg


TARGET_BLOCK = """\
    1   5   8  11      9    0.000     0.58576     3
    1   5   8  11      9  180.000     0.41840     2
;  --- other torsions about the same central bond 5-8 ---
    1   5   8   9      9    0.000     0.66944     3
    1   5   8  10      9    0.000     0.66944     3
    6   5   8  11      9    0.000     0.66944     3
    7   5   8  11      9    0.000     0.66944     3
    6   5   8   9      9    0.000     0.66944     3
    6   5   8  10      9    0.000     0.66944     3
    7   5   8   9      9    0.000     0.66944     3
    7   5   8  10      9    0.000     0.66944     3
"""

TYPED_BLOCK = """\
    1   5   8  11      9
;  --- other torsions about the same central bond 5-8 ---
    1   5   8   9      9
    1   5   8  10      9
    6   5   8  11      9
    7   5   8  11      9
    6   5   8   9      9
    6   5   8  10      9
    7   5   8   9      9
    7   5   8  10      9
"""


def _typed_topology(tmp_path, butane_itp, block=TYPED_BLOCK):
    """A butane .itp whose torsion set relies on [ dihedraltypes ] lookup.

    This is the normal shape of a real CGenFF or GAFF molecule topology, and
    it is why the pre-run refusal added for item 2 matters.
    """
    text = butane_itp.read_text()
    assert TARGET_BLOCK in text, "fixture layout changed; update TARGET_BLOCK"
    out = tmp_path / "typed.itp"
    out.write_text(text.replace(TARGET_BLOCK, block), newline="\n")
    return out


DIHEDRALTYPES_SECTION = """\
[ dihedraltypes ]
;   i      j      k      l  funct   phi0        kphi  mult
    X   CG321  CG321      X      9    0.0        20.0     3

[ moleculetype ]"""


@pytest.mark.needs_gmx
def test_dihedraltypes_entry_is_suppressed_by_explicit_params(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """Item 16 / spec 4.4: do not *trust* that explicit parameters suppress the
    [ dihedraltypes ] lookup -- check it.

    This butane variant carries a wildcard type entry that matches every
    quadruplet of the target torsion set, with a force constant (20 kJ/mol)
    thirty times the explicit one. If GROMACS applied it on top of the
    explicit line, or in place of the zeroed line, the section 4.5 residual
    would be enormous. It measures ~1e-5 kJ/mol, so the lookup is suppressed.
    """
    from ffrefine.config import Binaries

    text = butane_itp.read_text()
    assert "[ moleculetype ]" in text
    itp = tmp_path / "typed_ff.itp"
    itp.write_text(text.replace("[ moleculetype ]", DIHEDRALTYPES_SECTION, 1), newline="\n")

    s = read_pdb(butane_pdb)
    t = read_topology(itp)
    frames = rotated_frames(s, 8)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path
    )
    assert r.passed
    assert r.max_abs_error < 1e-3
    # The torsion really is being felt, i.e. the fixture is not vacuously flat.
    assert np.abs(r.analytic_difference).max() > 0.5

    # Direct proof of suppression: the type entry changes nothing at all.
    # Measured 2026-09-07 on GROMACS 2025.4 -- residual 9.7266e-06 kJ/mol with
    # and without the section, and bit-identical total potentials.
    plain = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), read_topology(butane_itp), (1, 5, 8, 11), s, frames, tmp_path / "p"
    )
    assert np.allclose(r.full_potential_kj, plain.full_potential_kj, atol=0.0, rtol=0.0)


SYSTEM_TAIL = """

[ system ]
butane in vacuum

[ molecules ]
BUT     1
"""


@pytest.mark.needs_gmx
def test_molecule_top_input_reaches_grompp(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """Item 4: spec section 2 says -p takes an .itp *or a .top*.

    Pre-fix the source was copied verbatim into full.itp and then wrapped, so
    the system declared BUT twice and grompp aborted with "number of
    coordinates in coordinate file (14) does not match topology (28)".
    """
    from ffrefine.config import Binaries

    top = tmp_path / "butane.top"
    top.write_text(butane_itp.read_text() + SYSTEM_TAIL, newline="\n")
    s = read_pdb(butane_pdb)
    t = read_topology(top)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, s.coords[None, :, :], tmp_path / "wd"
    )
    assert r.passed
    # And the same answer a plain .itp gives -- the tail was cosmetic.
    plain = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin),
        read_topology(butane_itp),
        (1, 5, 8, 11),
        s,
        s.coords[None, :, :],
        tmp_path / "plain",
    )
    assert r.full_potential_kj == pytest.approx(plain.full_potential_kj, abs=0.0, rel=0.0)


@pytest.mark.needs_gmx
def test_ff_include_supplies_missing_atomtypes(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """Item 4: make_system_top's `includes` parameter was wired to nothing.

    butane.itp is split into a force-field half ([ defaults ] + [ atomtypes ])
    and a molecule half. The molecule half alone cannot be grompp'd -- grompp
    rejects it with "Atomtype CG331 not found" -- so this passes only if
    --ff-include actually reaches the generated wrapper .top.
    """
    from ffrefine.config import Binaries

    text = butane_itp.read_text()
    head, _, tail = text.partition("[ moleculetype ]")
    assert tail, "fixture layout changed"
    ff = tmp_path / "ff.itp"
    ff.write_text(head, newline="\n")
    mol = tmp_path / "mol.itp"
    mol.write_text("[ moleculetype ]" + tail, newline="\n")

    s = read_pdb(butane_pdb)
    t = read_topology(mol)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin),
        t,
        (1, 5, 8, 11),
        s,
        s.coords[None, :, :],
        tmp_path / "wd",
        ff_includes=[str(ff)],
    )
    assert r.passed
    assert f'#include "{ff}"' in (tmp_path / "wd" / "full.top").read_text()


@pytest.mark.needs_gmx
def test_typed_butane_resolves_through_a_dihedraltypes_entry(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """Every term of T typed, matched by one wildcard entry: 9 quadruplets x 20 kJ/mol."""
    from ffrefine.config import Binaries

    text = _typed_topology(tmp_path, butane_itp).read_text()
    itp = tmp_path / "typed_with_types.itp"
    itp.write_text(text.replace("[ moleculetype ]", DIHEDRALTYPES_SECTION, 1), newline="\n")
    s = read_pdb(butane_pdb)
    t = read_topology(itp)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, rotated_frames(s, 8), tmp_path / "wd"
    )
    assert r.passed and r.max_abs_error < 1e-3
    assert np.abs(r.analytic_difference).max() > 50.0  # the 20 kJ/mol type is really felt


@pytest.mark.needs_gmx
def test_typed_butane_without_matching_types_fails_in_grompp(
    tmp_path, butane_pdb, butane_itp, gmx_bin
):
    """No [ dihedraltypes ] entry for the typed lines: grompp's own error, before any rerun."""
    from ffrefine.config import Binaries
    from ffrefine.errors import ExternalToolError

    s = read_pdb(butane_pdb)
    t = read_topology(_typed_topology(tmp_path, butane_itp))
    with pytest.raises(ExternalToolError):
        selfcheck.check_torsion_removal(
            Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, s.coords[None, :, :], tmp_path / "wd"
        )
    assert not (tmp_path / "wd" / "full").exists()  # no rerun was started


@pytest.mark.needs_gmx
def test_self_check_passes_on_pfoa_typed_torsions(tmp_path, gmx_bin, rotate_side):
    """The real CGenFF case: 20 terms about C1-C3, all from [ dihedraltypes ]."""
    from ffrefine.config import Binaries

    s = read_pdb(PFOA / "pfoa_gmx.pdb")
    t = read_topology(PFOA / "pfoa_gmx.top")
    # Turn the carboxyl group (atoms 1, 2, 25, 26) a full circle about C1-C3.
    frames = rotate_side(
        s.coords, t.bonds, fixed=3, moving=1, angles_deg=np.arange(0.0, 360.0, 30.0)
    )
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (25, 1, 3, 4), s, frames, tmp_path
    )
    assert r.passed and r.max_abs_error < 1e-3
    assert np.ptp(r.analytic_difference) > 1.0


@pytest.mark.needs_gmx
def test_self_check_passes_on_capecitabine_typed_torsions(tmp_path, gmx_bin, rotate_side):
    """Carbamate C7-N9, where two of the four typed lines resolve to no terms."""
    from ffrefine.config import Binaries

    s = read_pdb(CAPECITABINE / "capecitabine_gmx.pdb")
    t = read_topology(CAPECITABINE / "capecitabine_gmx.top")
    # Swing the pentyl carbamate side +-30 degrees; wider sweeps drive it into the rings.
    frames = rotate_side(
        s.coords, t.bonds, fixed=9, moving=7, angles_deg=np.linspace(-30.0, 30.0, 7)
    )
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (6, 7, 9, 10), s, frames, tmp_path
    )
    assert r.passed and r.max_abs_error < 1e-3
    assert np.ptp(r.analytic_difference) > 1.0
