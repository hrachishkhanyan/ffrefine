import pytest

from ffrefine.errors import ValidationError
from ffrefine.qm.xtb import scan_grid, write_scan_input


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
