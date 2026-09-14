import pytest
from pathlib import Path

import numpy as np

from ffrefine.errors import ParseError, ValidationError
from ffrefine.qm.xtb import parse_scan_log, scan_grid, write_scan_input

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
    assert "dihedral: 1,5,8,11,auto" in text
    assert "$scan" in text
    assert "1: 0.0,345.0,24" in text
    assert text.rstrip().endswith("$end")


def test_scan_input_adds_frozen_dihedrals_before_the_scan_block(tmp_path):
    """--freeze-dihedral holds other rotatable bonds; they must be constrained,
    not scanned, so they belong in $constrain and must not become scan entries."""
    text = write_scan_input(
        tmp_path / "scan.inp", (1, 5, 8, 11), 0.0, 345.0, 24, freeze=[(2, 1, 5, 8)]
    )
    constrain, scan = text.split("$scan")
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
