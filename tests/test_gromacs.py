import numpy as np
import pytest
from ffrefine.config import Binaries
from ffrefine.errors import ExternalToolError, ValidationError
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import make_system_top, write_zeroed_topology
from ffrefine.mm import gromacs

XVG = """\
# comment
@    title "Gromacs Energies"
@    xaxis  label "Time (ps)"
@ s0 legend "Bond"
@ s1 legend "Angle"
@ s2 legend "Proper Dih."
@ s3 legend "Potential"
    0.000000    1.500000    2.500000    3.500000    7.500000
    1.000000    1.600000    2.600000    3.600000    7.800000
"""


def test_parse_energy_xvg_names_and_values(tmp_path):
    p = tmp_path / "e.xvg"
    p.write_text(XVG)
    names, data = gromacs.parse_energy_xvg(p)
    assert names == ["Bond", "Angle", "Proper Dih.", "Potential"]
    assert data.shape == (2, 4)
    assert data[0, 3] == pytest.approx(7.5)
    assert data[1, 0] == pytest.approx(1.6)


def test_parse_energy_xvg_ignores_comments(tmp_path):
    p = tmp_path / "e.xvg"
    p.write_text(XVG)
    _, data = gromacs.parse_energy_xvg(p)
    assert np.all(np.isfinite(data))


def test_sp_mdp_is_vacuum_single_point():
    """The mdp must match spec section 4.4 exactly."""
    for line in (
        "integrator              = md",
        "nsteps                  = 0",
        "pbc                     = no",
        "rcoulomb                = 0",
        "rvdw                    = 0",
        "epsilon_r               = 1",
        "constraints             = none",
    ):
        assert line in gromacs.SP_MDP


def test_fallback_mdp_uses_large_box_cutoffs():
    assert "pbc                     = xyz" in gromacs.FALLBACK_MDP
    assert "rvdw                    = 3" in gromacs.FALLBACK_MDP
    assert "rcoulomb                = 3" in gromacs.FALLBACK_MDP


def test_energy_result_potential_lookup():
    r = gromacs.EnergyResult(
        terms={"Potential": np.array([1.0, 2.0]), "Bond": np.array([0.5, 0.5])},
        nframes=2,
        used_fallback=False,
    )
    assert r.potential() == pytest.approx([1.0, 2.0])


@pytest.mark.needs_gmx
def test_single_point_runs_on_butane(tmp_path, butane_pdb, butane_itp, gmx_bin):
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    frames = np.stack([s.coords, s.coords, s.coords])
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text())
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT")

    result = gromacs.single_point_energies(Binaries(gmx=gmx_bin), top, s, frames, tmp_path, "full")
    assert result.nframes == 3
    assert "Potential" in result.terms
    # identical frames must give identical energies
    assert result.potential()[0] == pytest.approx(result.potential()[1], abs=1e-6)

    # Butane has no impropers, so "Improper Dih." is the only requested term
    # legitimately absent; anything else missing would mean a silent
    # stdin-name mismatch (spec section 9.9) dropped a term into a
    # log.warning instead of failing loudly.
    assert set(gromacs.DEFAULT_TERMS) - set(result.terms) == {"Improper Dih."}

    # Potential must equal the sum of its components. This also catches a
    # legend/column misalignment in parse_energy_xvg and a Potential that is
    # wrong by a constant offset -- exactly what Tasks 13 and 21 inherit.
    # Verified empirically against GROMACS 2025.4: the residual is ~1e-6
    # kJ/mol, consistent with the .edr's single-precision storage.
    component_sum = sum(v for k, v in result.terms.items() if k != "Potential")
    assert component_sum == pytest.approx(result.potential(), abs=1e-3)


@pytest.mark.needs_gmx
def test_frame_count_matches_input(tmp_path, butane_pdb, butane_itp, gmx_bin):
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    rng = np.random.default_rng(5)
    frames = np.stack([s.coords + rng.normal(scale=0.002, size=s.coords.shape) for _ in range(7)])
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text())
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT")
    r = gromacs.single_point_energies(Binaries(gmx=gmx_bin), top, s, frames, tmp_path, "sp")
    assert r.nframes == 7
    assert r.potential().shape == (7,)


@pytest.mark.needs_gmx
def test_zeroing_lowers_proper_dihedral_energy(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """Sanity: the zeroed topology must have less Proper Dih. energy."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    frames = s.coords[None, :, :]
    t = read_topology(butane_itp)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")

    zero_itp = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), zero_itp)
    zero_top = tmp_path / "zeroed.top"
    make_system_top(zero_itp, zero_top, "BUT")

    b = Binaries(gmx=gmx_bin)
    ef = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "f", "full")
    ez = gromacs.single_point_energies(b, zero_top, s, frames, tmp_path / "z", "zero")
    assert ez.terms["Proper Dih."][0] < ef.terms["Proper Dih."][0]


# --------------------------------------------------------------------------
# cutoff guard (final review, item 6)
# --------------------------------------------------------------------------


def test_max_interatomic_distance_on_a_known_geometry():
    frames = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 1.0, 1.0]]])
    assert gromacs.max_interatomic_distance(frames) == pytest.approx(5.0)


def test_max_interatomic_distance_takes_the_worst_frame():
    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    assert gromacs.max_interatomic_distance(frames) == pytest.approx(7.0)


def test_max_interatomic_distance_handles_degenerate_input():
    assert gromacs.max_interatomic_distance(np.zeros((0, 5, 3))) == 0.0
    assert gromacs.max_interatomic_distance(np.zeros((3, 1, 3))) == 0.0


def test_cutoff_guard_accepts_a_molecule_that_fits():
    frames = np.zeros((2, 4, 3))
    frames[:, 1, 0] = 2.9
    gromacs._check_molecule_fits_cutoff(frames)  # must not raise


def test_cutoff_guard_refuses_a_molecule_that_does_not_fit():
    """Item 6: pairs beyond rvdw contribute nothing, and the spec 4.5
    self-check cannot see it -- the truncation is identical in the full and
    zeroed topologies and cancels in their difference."""
    frames = np.zeros((2, 4, 3))
    frames[1, 3, 0] = 3.5
    with pytest.raises(ValidationError) as excinfo:
        gromacs._check_molecule_fits_cutoff(frames)
    msg = str(excinfo.value)
    assert "3.500 nm" in msg
    assert "cannot detect it" in msg


def test_fallback_mdp_cutoffs_match_the_guard():
    """The mdp and the guard must use one number, not two that can drift."""
    for key in ("rlist", "rcoulomb", "rvdw"):
        line = next(ln for ln in gromacs.FALLBACK_MDP.splitlines() if ln.startswith(key))
        assert float(line.split("=")[1]) == pytest.approx(gromacs.FALLBACK_CUTOFF_NM)


@pytest.mark.needs_gmx
def test_oversized_molecule_is_refused_before_mdrun(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """End to end: the guard fires on the fallback path and no energies are
    produced that a caller could mistake for correct ones."""
    s = read_pdb(butane_pdb)
    top = read_topology(butane_itp)
    system = tmp_path / "topol.top"
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text(), newline="\n")
    make_system_top(itp, system, top.name)

    stretched = s.coords.copy()
    stretched[7:] += np.array([4.0, 0.0, 0.0])  # C3 side pushed 4 nm away
    frames = np.stack([s.coords, stretched])

    with pytest.raises(ValidationError, match="does not fit inside the MM cutoff"):
        gromacs.single_point_energies(
            Binaries(gmx=gmx_bin), system, s, frames, tmp_path / "wd", "sp"
        )
    assert not (tmp_path / "wd" / "sp.edr").exists()
