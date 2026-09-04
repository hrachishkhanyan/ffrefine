import numpy as np
import pytest
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import make_system_top, write_refit_topology, write_zeroed_topology
from ffrefine.mm.torsionset import collect_torsion_set


def test_zeroed_topology_removes_all_nine(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    removed = write_zeroed_topology(t, (1, 5, 8, 11), out)
    assert len(removed) == 9
    z = read_topology(out)
    tset = collect_torsion_set(z, (1, 5, 8, 11))
    assert len(tset) == 9
    for d in tset:
        assert d.params[1] == pytest.approx(0.0)


def test_zeroed_lines_use_funct_9_with_explicit_zeros(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    for d in collect_torsion_set(z, (1, 5, 8, 11)):
        assert d.funct == 9
        assert len(d.params) == 3
        assert d.params == pytest.approx([0.0, 0.0, 1.0])


def test_input_file_is_not_modified(tmp_path, butane_itp):
    before = butane_itp.read_text()
    t = read_topology(butane_itp)
    write_zeroed_topology(t, (1, 5, 8, 11), tmp_path / "z.itp")
    assert butane_itp.read_text() == before


def test_other_sections_survive_untouched(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    assert z.natoms == 14
    assert len(z.bonds) == 13
    assert len(z.pairs) == 27
    assert len(z.angles) == 24
    assert "[ atomtypes ]" in out.read_text()


def test_pairs_are_never_removed(tmp_path, butane_itp):
    """Spec section 9.2: 1-4 interactions must survive in both topologies."""
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    assert read_topology(out).pairs == t.pairs


def test_methyl_torsions_survive_zeroing(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    methyl = [d for d in z.propers if d.central_bond == (1, 5)]
    assert len(methyl) == 9
    assert all(d.params[1] > 0 for d in methyl)


def test_impropers_survive_zeroing(tmp_path, butane_itp):
    """Impropers (funct 2/4) are never modified (spec section 9.4), even when
    they happen to share atom indices and a central bond with a zeroed proper.
    """
    text = butane_itp.read_text().replace(
        "[ dihedrals ]\n",
        "[ dihedrals ]\n"
        "    1   5   8  11      4    0.000    10.00000     2  ; synthetic improper\n",
    )
    src = tmp_path / "with_improper.itp"
    src.write_text(text)
    t = read_topology(src)
    assert len(t.impropers) == 1

    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    assert len(z.impropers) == 1
    assert z.impropers[0] == t.impropers[0]


def test_refit_topology_writes_series_on_target_only(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "refit.itp"
    write_refit_topology(
        t,
        (1, 5, 8, 11),
        np.array([1.25, 0.4]),
        np.array([0.0, 180.0]),
        np.array([3, 2]),
        out,
        distribute=False,
    )
    r = read_topology(out)
    target = [d for d in r.propers if (d.ai, d.aj, d.ak, d.al) in ((1, 5, 8, 11), (11, 8, 5, 1))]
    assert len(target) == 2
    assert {int(d.params[2]) for d in target} == {2, 3}
    others = [
        d
        for d in collect_torsion_set(r, (1, 5, 8, 11))
        if (d.ai, d.aj, d.ak, d.al) not in ((1, 5, 8, 11), (11, 8, 5, 1))
    ]
    assert all(d.params[1] == pytest.approx(0.0) for d in others)


def test_refit_distribute_divides_by_set_size(tmp_path, butane_itp):
    """Spec section 9.7: with --distribute, k_n is divided by N_T."""
    t = read_topology(butane_itp)
    out = tmp_path / "refit.itp"
    write_refit_topology(
        t,
        (1, 5, 8, 11),
        np.array([9.0]),
        np.array([0.0]),
        np.array([3]),
        out,
        distribute=True,
    )
    r = read_topology(out)
    tset = collect_torsion_set(r, (1, 5, 8, 11))
    assert len(tset) == 9
    for d in tset:
        assert d.params[1] == pytest.approx(1.0)  # 9.0 / 9 members


def test_make_system_top(tmp_path, butane_itp):
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text())
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT")
    text = top.read_text()
    assert "mol.itp" in text
    assert "[ molecules ]" in text
    assert "BUT" in text
