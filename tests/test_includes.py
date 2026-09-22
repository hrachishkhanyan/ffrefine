"""Include handling for the topologies ffrefine writes (milestone 6a design, section 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from ffrefine.errors import ParseError, ValidationError
from ffrefine.io import includes
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX

LINES = [
    "; molecule",
    '#include "./charmm36.ff/forcefield.itp"',
    '# include "./charmm36.ff/mol_ffbonded.itp"',
    '#include "posre.itp"',
    "#include <gmxlib.itp>",
    "[ moleculetype ]",
]


def _user_dir(tmp_path: Path) -> Path:
    """A user directory holding a force-field folder the LINES include."""
    user = tmp_path / "user"
    (user / "charmm36.ff").mkdir(parents=True)
    (user / "charmm36.ff" / "forcefield.itp").write_text("[ defaults ]\n1 2 yes 1.0 1.0\n")
    (user / "charmm36.ff" / "mol_ffbonded.itp").write_text("")
    return user


def test_absolutise_rewrites_only_relative_includes_that_exist(tmp_path):
    user = _user_dir(tmp_path)
    ff = (user / "charmm36.ff").resolve()
    out = includes.absolutise_includes(LINES, user)
    assert out[1] == f'#include "{ff / "forcefield.itp"}"'
    assert out[2] == f'# include "{ff / "mol_ffbonded.itp"}"'
    assert out[3] == '#include "posre.itp"'  # not under the base: grompp searches for it
    assert out[4] == "#include <gmxlib.itp>"  # angle brackets are never rewritten
    assert out[0] == LINES[0] and out[5] == LINES[5]


def test_absolutise_leaves_an_absolute_include_alone(tmp_path):
    user = _user_dir(tmp_path)
    line = f'#include "{(user / "charmm36.ff" / "forcefield.itp").resolve()}"'
    assert includes.absolutise_includes([line], tmp_path / "elsewhere") == [line]


def test_absolutise_keeps_a_trailing_comment(tmp_path):
    user = _user_dir(tmp_path)
    out = includes.absolutise_includes(['#include "./charmm36.ff/forcefield.itp" ; ff'], user)
    assert out[0].endswith('forcefield.itp" ; ff')


def test_mark_prepends_the_base_when_a_relative_include_resolved(tmp_path):
    user = _user_dir(tmp_path)
    out = includes.mark_include_base(LINES, user)
    assert out[:2] == [MARKER_NOTE, f"{MARKER_PREFIX}{user.resolve()}"]
    assert out[2:] == LINES  # the include lines themselves are untouched


def test_mark_writes_nothing_without_a_resolvable_relative_include(tmp_path):
    lines = ["[ moleculetype ]", '#include "posre.itp"']
    assert includes.mark_include_base(lines, tmp_path) == lines


def test_mark_replaces_an_existing_marker_instead_of_adding_a_second(tmp_path):
    user = _user_dir(tmp_path)
    once = includes.mark_include_base(LINES, user)
    assert includes.mark_include_base(once, user) == once


def test_absolutise_strips_a_marker(tmp_path):
    user = _user_dir(tmp_path)
    out = includes.absolutise_includes(includes.mark_include_base(LINES, user), user)
    assert not any(line == MARKER_NOTE or line.startswith(MARKER_PREFIX) for line in out)


def test_apply_include_mode_dispatches(tmp_path):
    user = _user_dir(tmp_path)
    assert includes.apply_include_mode(LINES, user, "absolute") == includes.absolutise_includes(
        LINES, user
    )
    assert includes.apply_include_mode(LINES, user, "verbatim") == includes.mark_include_base(
        LINES, user
    )
    with pytest.raises(ValueError):
        includes.apply_include_mode(LINES, user, "relative")


def test_include_base_is_the_file_directory_without_a_marker(tmp_path):
    user = _user_dir(tmp_path)
    top = user / "mol.top"
    top.write_text("\n".join(LINES) + "\n")
    assert includes.include_base(top) == user.resolve()


def test_include_base_follows_the_marker(tmp_path):
    user = _user_dir(tmp_path)
    moved = tmp_path / "out" / "04_charges"
    moved.mkdir(parents=True)
    copy = moved / "molecule_charged.itp"
    copy.write_text("\n".join(includes.mark_include_base(LINES, user)) + "\n")
    assert includes.include_base(copy) == user.resolve()


def test_a_marker_naming_a_missing_directory_is_refused(tmp_path):
    copy = tmp_path / "x.itp"
    copy.write_text(f"{MARKER_PREFIX}{tmp_path / 'gone'}\n[ moleculetype ]\n")
    with pytest.raises(ValidationError, match="does not exist"):
        includes.include_base(copy)


def test_two_different_markers_are_refused(tmp_path):
    copy = tmp_path / "x.itp"
    copy.write_text(f"{MARKER_PREFIX}/a\n{MARKER_PREFIX}/b\n")
    with pytest.raises(ParseError, match="two ffrefine include-base markers"):
        includes.include_base(copy)


def test_has_includes():
    assert includes.has_includes(['# include "a.itp"'])
    assert not includes.has_includes(["; #include in a comment", "[ atoms ]"])
