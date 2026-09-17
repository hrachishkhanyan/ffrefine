import pytest
from pathlib import Path

import numpy as np

from ffrefine.config import Binaries
from ffrefine.errors import ExternalToolError, ParseError, ValidationError
from ffrefine.io.structure import read_pdb
from ffrefine.qm.xtb import (
    derive_charge,
    parse_scan_log,
    run_scan,
    scan_grid,
    write_scan_input,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "butane" / "xtb"


@pytest.fixture
def forward_log() -> Path:
    return FIXTURES / "xtbscan_forward_4pt.log"


def test_scan_grid_excludes_the_duplicated_endpoint():
    """Spec 4.3: span 0 to 360-step, so 0 and 360 are not both sampled."""
    start, end, n = scan_grid(15.0)
    assert (start, end, n) == (0.0, 345.0, 24)


def test_scan_grid_rejects_a_step_that_does_not_divide_a_turn():
    with pytest.raises(ValidationError, match="360"):
        scan_grid(7.0)


def test_scan_grid_rejects_a_nonpositive_step():
    with pytest.raises(ValidationError):
        scan_grid(0.0)


def test_scan_input_matches_the_spec_block(tmp_path):
    text = write_scan_input(tmp_path / "scan.inp", (1, 5, 8, 11), 0.0, 345.0, 24)
    assert "$constrain" in text
    assert "force constant=1.0" in text
    assert "dihedral: 1,5,8,11,0.0" in text
    assert "$scan" in text
    assert "1: 0.0,345.0,24" in text
    assert text.rstrip().endswith("$end")


def test_scanned_dihedral_states_the_start_angle_rather_than_auto(tmp_path):
    """``auto`` takes the constraint from the input geometry, not from the grid.

    When the two disagree, xtb honours ``auto`` for the first frame and ramps
    from there, so the requested start point is never sampled and the input
    geometry's angle is sampled twice.  Verified against xtb 6.7.1: butane
    (anti, 180 degrees) scanned 0 to 345 returned a forward leg whose first
    frame measured 180.00 and whose thirteenth measured 179.92 -- 0.076 degrees
    apart.  That duplicated abscissa is what spec 4.3's "do not sample both
    endpoints" rule exists to prevent, and it makes the forward and reverse
    legs impossible to pair by angle.
    """
    text = write_scan_input(tmp_path / "scan.inp", (1, 5, 8, 11), 180.0, 165.0, 24)

    assert "  dihedral: 1,5,8,11,180.0" in text
    assert "1,5,8,11,auto" not in text


def test_scan_input_adds_frozen_dihedrals_before_the_scan_block(tmp_path):
    """--freeze-dihedral holds other rotatable bonds; they must be constrained,
    not scanned, so they belong in $constrain and must not become scan entries."""
    text = write_scan_input(
        tmp_path / "scan.inp", (1, 5, 8, 11), 0.0, 345.0, 24, freeze=[(2, 1, 5, 8)]
    )
    constrain, scan = text.split("$scan")
    # A frozen dihedral is held wherever the input geometry put it, which is
    # exactly what auto means; only the scanned one needs an explicit value.
    assert "dihedral: 2,1,5,8,auto" in constrain
    assert "2,1,5,8" not in scan
    # Exactly one scan entry, numbered 1, referring to the target dihedral.
    assert scan.count(":") == 1


def test_scan_input_is_written_with_lf(tmp_path):
    path = tmp_path / "scan.inp"
    write_scan_input(path, (1, 5, 8, 11), 0.0, 345.0, 24)
    assert b"\r\n" not in path.read_bytes()


def test_scan_input_rejects_a_repeated_atom(tmp_path):
    with pytest.raises(ValidationError):
        write_scan_input(tmp_path / "scan.inp", (1, 5, 5, 11), 0.0, 345.0, 24)


def test_scan_input_emits_target_before_freeze_dihedrals(tmp_path):
    """$scan's ``1:`` addresses the first dihedral constraint in $constrain.
    If the target quad's constraint line were emitted after freeze lines,
    reordering would silently scan the wrong bond with exit 0, producing
    a scientifically wrong result that looks correct to automated checks.
    Assert the target appears before every freeze entry."""
    text = write_scan_input(
        tmp_path / "scan.inp",
        (1, 5, 8, 11),
        0.0,
        345.0,
        24,
        freeze=[(2, 6, 7, 9), (3, 4, 5, 10)],
    )
    constrain, _ = text.split("$scan")
    lines = constrain.split("\n")
    dihedral_lines = [i for i, line in enumerate(lines) if "dihedral:" in line]
    # First dihedral line is the target (1,5,8,11); the rest are freeze entries.
    assert "1,5,8,11" in lines[dihedral_lines[0]]
    assert "2,6,7,9" in lines[dihedral_lines[1]]
    assert "3,4,5,10" in lines[dihedral_lines[2]]


def test_parse_scan_log_reads_every_frame(forward_log):
    frames_nm, energy_kj = parse_scan_log(forward_log)
    assert frames_nm.shape == (4, 14, 3)
    assert energy_kj.shape == (4,)


def test_parse_scan_log_converts_hartree_to_kj(forward_log):
    """First frame's recorded energy is -13.665127531525 Eh."""
    _, energy_kj = parse_scan_log(forward_log)
    assert energy_kj[0] == pytest.approx(-13.665127531525 * 2625.499639, rel=1e-12)


def test_parse_scan_log_converts_angstrom_to_nm(forward_log):
    """The log's first atom is at x = -1.92585140500682 Angstrom."""
    frames_nm, _ = parse_scan_log(forward_log)
    assert frames_nm[0][0][0] == pytest.approx(-0.192585140500682, abs=1e-12)


def test_parse_scan_log_recovers_the_butane_profile(forward_log):
    """Sanity: relative energies must reproduce the probe's kJ/mol profile."""
    _, energy_kj = parse_scan_log(forward_log)
    relative = energy_kj - energy_kj.min()
    assert relative == pytest.approx([0.0, 10.741, 10.740, 20.691], abs=0.01)


def test_parse_scan_log_rejects_a_truncated_frame(tmp_path):
    """A killed xtb leaves a partial frame; that must not parse as valid."""
    path = tmp_path / "xtbscan.log"
    path.write_text("3\n energy: -1.0 xtb: 6.7.1\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n", newline="\n")
    with pytest.raises(ParseError, match="truncated|incomplete"):
        parse_scan_log(path)


def test_parse_scan_log_rejects_a_missing_energy_comment(tmp_path):
    path = tmp_path / "xtbscan.log"
    path.write_text("1\nno energy here\nC 0.0 0.0 0.0\n", newline="\n")
    with pytest.raises(ParseError, match="energy"):
        parse_scan_log(path)


def test_parse_scan_log_rejects_a_varying_atom_count(tmp_path):
    """Frames must all describe the same molecule."""
    path = tmp_path / "xtbscan.log"
    path.write_text(
        "1\n energy: -1.0 xtb: x\nC 0.0 0.0 0.0\n"
        "2\n energy: -2.0 xtb: x\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n",
        newline="\n",
    )
    with pytest.raises(ParseError, match="atom count"):
        parse_scan_log(path)


def test_parse_scan_log_rejects_an_empty_file(tmp_path):
    path = tmp_path / "xtbscan.log"
    path.write_text("", newline="\n")
    with pytest.raises(ParseError):
        parse_scan_log(path)


def test_run_scan_without_xtb_names_the_override_flag(tmp_path, butane_pdb):
    s = read_pdb(butane_pdb)
    with pytest.raises(Exception, match="xtb-bin"):
        run_scan(
            Binaries(xtb=None),
            s,
            (1, 5, 8, 11),
            tmp_path,
            start_deg=0.0,
            end_deg=345.0,
            npoints=24,
            charge=0,
            uhf=0,
        )


def test_run_scan_refuses_when_no_scan_log_appears(tmp_path, butane_pdb, monkeypatch):
    """If the $scan block were rejected, xtb can exit 0 having written nothing.

    That is the spec 4.3 failure this check exists for: a silent no-op must not
    be mistaken for a scan.
    """
    import ffrefine.qm.xtb as mod

    class _Proc:
        stdout = "normal termination of xtb"
        stderr = ""
        returncode = 0

    monkeypatch.setattr(mod, "run_command", lambda *a, **k: _Proc())
    s = read_pdb(butane_pdb)
    with pytest.raises(ExternalToolError, match=r"\$scan|xtbscan.log"):
        run_scan(
            Binaries(xtb="/usr/bin/true"),
            s,
            (1, 5, 8, 11),
            tmp_path,
            start_deg=0.0,
            end_deg=345.0,
            npoints=24,
            charge=0,
            uhf=0,
        )


def test_run_scan_refuses_a_short_scan(tmp_path, butane_pdb, monkeypatch):
    """24 points requested, 4 returned, means the scan did not do what was asked."""
    import shutil as _shutil

    import ffrefine.qm.xtb as mod

    class _Proc:
        stdout = "RELAXED SCAN\nnormal termination of xtb"
        stderr = ""
        returncode = 0

    def _fake(argv, workdir, log_stem, **kwargs):
        _shutil.copy(FIXTURES / "xtbscan_forward_4pt.log", Path(workdir) / "xtbscan.log")
        return _Proc()

    monkeypatch.setattr(mod, "run_command", _fake)
    s = read_pdb(butane_pdb)
    with pytest.raises(ExternalToolError, match="24"):
        run_scan(
            Binaries(xtb="/usr/bin/true"),
            s,
            (1, 5, 8, 11),
            tmp_path,
            start_deg=0.0,
            end_deg=345.0,
            npoints=24,
            charge=0,
            uhf=0,
        )


def test_run_scan_refuses_identical_frames(tmp_path, butane_pdb, monkeypatch):
    """R44: xtb can stall the relaxed-scan optimiser (seen under -P > 1 in this
    environment) and still exit 0, report RELAXED SCAN, and write a log with
    exactly the requested frame count -- every existing check passes even
    though the geometry never moved. A real relaxed scan moves atoms at every
    grid point by construction, so N identical frames is direct proof the
    scan did not run and must be refused, not returned as data.
    """
    import shutil as _shutil

    import ffrefine.qm.xtb as mod

    class _Proc:
        stdout = "RELAXED SCAN\nnormal termination of xtb"
        stderr = ""
        returncode = 0

    def _fake(argv, workdir, log_stem, **kwargs):
        # Four copies of the fixture's first frame: same atom count and
        # frame count as requested below, but zero geometric motion.
        first_frame = "\n".join(
            (FIXTURES / "xtbscan_forward_4pt.log").read_text().splitlines()[0:16]
        )
        stalled_log = "\n".join([first_frame] * 4) + "\n"
        (Path(workdir) / "xtbscan.log").write_text(stalled_log, newline="\n")
        return _Proc()

    monkeypatch.setattr(mod, "run_command", _fake)
    s = read_pdb(butane_pdb)
    with pytest.raises(ExternalToolError, match="identical|stall"):
        run_scan(
            Binaries(xtb="/usr/bin/true"),
            s,
            (1, 5, 8, 11),
            tmp_path,
            start_deg=0.0,
            end_deg=360.0,
            npoints=4,
            charge=0,
            uhf=0,
        )


@pytest.mark.needs_xtb
def test_run_scan_drives_real_xtb(tmp_path, butane_pdb, xtb_bin):
    """Spec 4.3's runtime verification, as a test rather than as prose.

    Four points keeps this near 15 s; the default 24-point grid takes ~80 s.

    nthreads=1, not the brief's 4: on this xtb 6.7.1 build under WSL2, ``-P 4``
    makes the relaxed-scan optimizer silently stall (every point converges
    back to the pre-scan geometry, span 0.0) in roughly half of repeated
    runs, while -P 1 and -P 2 were reliable across every trial taken. That is
    exactly the quiet no-op spec 4.3 warns about, just triggered by xtb's own
    threading rather than by a rejected $scan block -- so it is precisely
    what this test's span assertion exists to catch, not a reason to weaken
    it. See task-4-report.md for the reproduction data.
    """
    s = read_pdb(butane_pdb)
    frames_nm, energy_kj = run_scan(
        Binaries(xtb=xtb_bin),
        s,
        (1, 5, 8, 11),
        tmp_path,
        start_deg=0.0,
        end_deg=270.0,
        npoints=4,
        charge=0,
        uhf=0,
        nthreads=1,
    )
    assert frames_nm.shape == (4, s.natoms, 3)
    assert energy_kj.shape == (4,)
    # Butane's syn barrier is real chemistry, not a smoke test.  The grid runs
    # to 270, not 360: at 0-360 over four points the endpoints are the same
    # angle, so syn is sampled twice and the anti minimum -- the bottom of the
    # very barrier being measured -- is never sampled at all.  Measured angles
    # here are 0, 90, -180, -90 and the span is about 20.7 kJ/mol.
    span = energy_kj.max() - energy_kj.min()
    assert 14.0 < span < 26.0


def test_derive_charge_reads_the_topology_sum(butane_itp):
    from ffrefine.io.topology import read_topology

    assert derive_charge(read_topology(butane_itp)) == 0


def test_derive_charge_rounds_a_near_integer_sum(butane_itp):
    """CGenFF charges print to 3-4 decimals, so the sum is rarely exact."""
    from ffrefine.io.topology import read_topology

    top = read_topology(butane_itp)
    top.atoms[0].charge += 0.999
    assert derive_charge(top) == 1


def test_derive_charge_handles_a_negative_ion(butane_itp):
    """Rounding must work on both sides of zero, not just upward."""
    from ffrefine.io.topology import read_topology

    top = read_topology(butane_itp)
    top.atoms[0].charge -= 1.002
    assert derive_charge(top) == -1


def test_derive_charge_refuses_a_sum_that_is_not_near_an_integer(butane_itp):
    from ffrefine.io.topology import read_topology

    top = read_topology(butane_itp)
    top.atoms[0].charge += 0.4
    with pytest.raises(ValidationError, match="0.4"):
        derive_charge(top)


def test_derive_charge_refusal_names_the_escape_hatch(butane_itp):
    """A user blocked by this needs to be told how to proceed, not just stopped."""
    from ffrefine.io.topology import read_topology

    top = read_topology(butane_itp)
    top.atoms[0].charge += 0.4
    with pytest.raises(ValidationError, match="--charge"):
        derive_charge(top)


def test_derive_charge_override_wins(butane_itp):
    from ffrefine.io.topology import read_topology

    assert derive_charge(read_topology(butane_itp), override=-1) == -1


def test_derive_charge_override_wins_even_over_a_refusable_sum(butane_itp):
    """The override is the escape hatch the refusal advertises, so it must not re-check."""
    from ffrefine.io.topology import read_topology

    top = read_topology(butane_itp)
    top.atoms[0].charge += 0.4
    assert derive_charge(top, override=0) == 0


@pytest.mark.needs_xtb
def test_optimize_geometry_moves_the_structure_and_keeps_the_atom_order(
    tmp_path, butane_pdb, xtb_bin
):
    from ffrefine.qm.xtb import optimize_geometry

    s = read_pdb(butane_pdb)
    out = optimize_geometry(
        Binaries(xtb=xtb_bin), s, s.coords, tmp_path, charge=0, uhf=0, nthreads=1
    )

    assert out.shape == s.coords.shape
    # The hand-authored fixture is not at an xtb minimum, so it must move,
    # but it must stay the same molecule rather than fly apart.
    assert np.abs(out - s.coords).max() > 1e-4
    assert np.abs(out - s.coords).max() < 0.1


def test_optimize_geometry_refuses_output_that_never_converged(tmp_path, butane_pdb, monkeypatch):
    """xtb can exit 0 having given up; the banner is the real signal."""
    import ffrefine.qm.xtb as xtb_module
    from ffrefine.errors import ExternalToolError

    class _Proc:
        stdout = "some output without the magic words\n"
        stderr = ""

    monkeypatch.setattr(xtb_module, "run_command", lambda *a, **k: _Proc())
    s = read_pdb(butane_pdb)

    with pytest.raises(ExternalToolError, match="converge"):
        xtb_module.optimize_geometry(
            Binaries(xtb="/nonexistent/xtb"), s, s.coords, tmp_path, charge=0
        )
