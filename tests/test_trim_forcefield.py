"""scripts/trim_forcefield.py -- the fixture force-field trimmer (design section 8).

The script is not a package module, so it is loaded from its path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "trim_forcefield.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_tpr_dumps.py"
PFOA = ROOT / "fixtures" / "pfoa_gromacs"
CAPECITABINE = ROOT / "fixtures" / "capecitabine_gromacs"
PFOA_DUMP = PFOA / "gromacs" / "pfoa_dump_2025.4.txt"


@pytest.fixture(scope="module")
def trim():
    spec = importlib.util.spec_from_file_location("trim_forcefield", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compare_tpr_dumps():
    spec = importlib.util.spec_from_file_location("compare_tpr_dumps", COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keeps_entries_whose_types_are_all_used_or_wildcard(trim):
    lines = [
        "[ dihedraltypes ]",
        "; i j k l",
        "X CG321 CG321 X 9 0.0 1.0 3",
        "CG321 CG321 CG321 CG321 9 0.0 1.0 3",
        "CG321 CG321 CG321 HGA2 9 0.0 1.0 3",
    ]
    assert trim.trim_lines(lines, {"CG321"}) == lines[:4]


def test_counts_type_fields_per_section(trim):
    """A pairtypes line has two type fields; its numbers are never read as types."""
    lines = ["[ pairtypes ]", "CG321 CG321 1 0.3 0.2", "CG321 HGA2 1 0.3 0.2"]
    assert trim.trim_lines(lines, {"CG321"}) == lines[:2]


def test_drops_a_continued_cmap_entry_whole(trim):
    lines = [
        "[ cmaptypes ]",
        "",
        "C NH1 CT1 C NH1 1 2 2\\",
        "1.0 2.0\\",
        "3.0 4.0",
        "",
        "CG321 CG321 CG321 CG321 CG321 1 2 2\\",
        "5.0 6.0\\",
        "7.0 8.0",
    ]
    assert trim.trim_lines(lines, {"CG321"}) == ["[ cmaptypes ]", "", "", *lines[6:]]


def test_copies_directives_comments_and_other_sections(trim):
    lines = [
        "[ atomtypes ]",
        "#ifdef HEAVY_H",
        "HT 1 4.032 0.000 A 0.0 0.0",
        "#else",
        "HT 1 1.008 0.000 A 0.0 0.0",
        "#endif",
        "; a comment",
        "[ moleculetype ]",
        "SOL 2",
    ]
    assert trim.trim_lines(lines, set()) == [lines[0], lines[1], lines[3], *lines[5:]]


def test_used_types_include_water_and_ions(trim):
    used = trim.used_types(PFOA / "pfoa_gmx.top", PFOA / "charmm36.ff", "charmm36.ff")
    assert {"CG2O2", "OG311", "FGA2", "HGP1"} <= used
    assert {"OT", "HT", "SOD", "CLA"} <= used


@pytest.mark.parametrize(
    "mol, top", [(PFOA, "pfoa_gmx.top"), (CAPECITABINE, "capecitabine_gmx.top")]
)
def test_trimming_a_committed_fixture_changes_nothing(trim, tmp_path, mol, top):
    """The committed trees are this script's output, and trimming is idempotent."""
    out = tmp_path / "charmm36.ff"
    trim.trim_forcefield(mol / "charmm36.ff", mol / top, out)
    committed = sorted(p.name for p in (mol / "charmm36.ff").iterdir())
    assert sorted(p.name for p in out.iterdir()) == committed
    for name in committed:
        assert (out / name).read_bytes() == (mol / "charmm36.ff" / name).read_bytes(), name


def test_compare_tpr_dumps_identical_dump_returns_0(compare_tpr_dumps):
    dump = str(PFOA_DUMP)
    assert compare_tpr_dumps.main([dump, dump]) == 0


def test_compare_tpr_dumps_changed_force_constant_returns_1(compare_tpr_dumps, tmp_path, capsys):
    original = PFOA_DUMP.read_text()
    changed = original.replace("cpA= 2.88696003e+00", "cpA= 2.88696004e+00", 1)
    assert changed != original
    copy = tmp_path / "pfoa_dump_2025.4_changed.txt"
    copy.write_text(changed)
    assert compare_tpr_dumps.main([str(PFOA_DUMP), str(copy)]) == 1
    captured = capsys.readouterr()
    assert any("2.88696004" in line for line in captured.out.splitlines())
