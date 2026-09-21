import numpy as np
import pytest

from ffrefine.charges.equivalence import average_within_groups, equivalent_hydrogen_groups
from ffrefine.charges.esp import inverse_distances, normal_equations, potential_from_charges
from ffrefine.charges.grid import mk_grid
from ffrefine.charges.resp import RESP_A_STAGE1, RESP_B, resp_fit
from ffrefine.config import discover_binaries
from ffrefine.errors import ValidationError
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.validate import adopt_topology_naming, element_of
from ffrefine.qm.orca import parse_chelpg_charges, run_esp_job, run_vpot
from ffrefine.qm.xtb import derive_charge
from ffrefine.units import nm_to_bohr


def _case(seed: int = 0, natoms: int = 5):
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(natoms, 3)) * 2.0
    grid = rng.normal(size=(900, 3)) * 9.0
    return atoms, grid


def test_an_unrestrained_fit_recovers_the_charges_that_made_the_potential():
    """The load-bearing test: a transposed matrix or a wrong sign dies here."""
    atoms, grid = _case()
    truth = np.array([0.4, -0.3, 0.25, -0.2, -0.15])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    out = resp_fit(
        inv_r,
        v,
        restrain=np.zeros(5, dtype=bool),
        groups=[],
        stage2_free=[],
        total_charge=float(truth.sum()),
        a_stage1=0.0,
        a_stage2=0.0,
    )

    assert out.charges == pytest.approx(truth, abs=1e-8)
    assert out.rrms == pytest.approx(0.0, abs=1e-10)


def test_the_total_charge_constraint_is_exact():
    atoms, grid = _case(seed=1)
    truth = np.array([0.4, -0.3, 0.25, -0.2, -0.15])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    out = resp_fit(
        inv_r,
        v,
        restrain=np.ones(5, dtype=bool),
        groups=[],
        stage2_free=[],
        total_charge=-1.0,
    )

    assert out.charges.sum() == pytest.approx(-1.0, abs=1e-10)


def test_the_restraint_pulls_heavy_atom_charges_toward_zero():
    """That is the whole purpose of RESP over a plain ESP fit."""
    atoms, grid = _case(seed=2)
    truth = np.array([0.9, -0.8, 0.5, -0.4, -0.2])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)
    common = dict(groups=[], stage2_free=[], total_charge=float(truth.sum()))

    free = resp_fit(inv_r, v, restrain=np.zeros(5, dtype=bool), a_stage1=0.0, **common)
    tight = resp_fit(inv_r, v, restrain=np.ones(5, dtype=bool), a_stage1=0.01, **common)

    assert np.abs(tight.charges).sum() < np.abs(free.charges).sum()


def test_stage_two_makes_an_equivalence_group_exactly_equal():
    atoms, grid = _case(seed=3)
    truth = np.array([-0.4, 0.1, 0.12, 0.09, 0.09])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    out = resp_fit(
        inv_r,
        v,
        restrain=np.array([True, False, False, False, False]),
        groups=[[1, 2, 3]],
        stage2_free=[0, 1, 2, 3],
        total_charge=float(truth.sum()),
    )

    assert out.charges[1] == pytest.approx(out.charges[2], abs=1e-10)
    assert out.charges[2] == pytest.approx(out.charges[3], abs=1e-10)
    assert out.charges.sum() == pytest.approx(truth.sum(), abs=1e-10)


def test_atoms_outside_stage_two_keep_their_stage_one_values():
    atoms, grid = _case(seed=4)
    truth = np.array([-0.4, 0.1, 0.12, 0.09, 0.09])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)
    common = dict(restrain=np.ones(5, dtype=bool), total_charge=float(truth.sum()))

    stage1_only = resp_fit(inv_r, v, groups=[], stage2_free=[], **common)
    both = resp_fit(inv_r, v, groups=[[1, 2]], stage2_free=[1, 2], **common)

    assert both.charges[4] == pytest.approx(stage1_only.charges[4], abs=1e-10)


def test_a_group_not_contained_in_stage2_free_is_refused():
    """The usual way this happens: 1-based groups passed without converting."""
    atoms, grid = _case(seed=6)
    truth = np.array([-0.4, 0.1, 0.12, 0.09, 0.09])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    with pytest.raises(ValidationError, match="stage2_free"):
        resp_fit(
            inv_r,
            v,
            restrain=np.array([True, False, False, False, False]),
            groups=[[1, 2, 3]],
            stage2_free=[1, 2],  # atom 3 is missing
            total_charge=float(truth.sum()),
        )


def test_groups_with_an_empty_stage2_free_is_refused():
    """Otherwise the group would vanish silently instead of constraining anything."""
    atoms, grid = _case(seed=7)
    truth = np.array([-0.4, 0.1, 0.12, 0.09, 0.09])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    with pytest.raises(ValidationError, match="stage2_free"):
        resp_fit(
            inv_r,
            v,
            restrain=np.array([True, False, False, False, False]),
            groups=[[1, 2, 3]],
            stage2_free=[],
            total_charge=float(truth.sum()),
        )


def test_stage_one_satisfies_the_kkt_stationarity_condition():
    """Pins the restraint's exact form: convention, the sqrt, and the mask.

    Minimising ``1/2 sum (V - Phi q)^2 + a sum restrain * (sqrt(q^2+b^2) - b)``
    subject only to a total-charge constraint means the Lagrangian's gradient
    vanishes at the optimum: ``A q - B + restrain * a * q / sqrt(q^2 + b^2)``
    must be the same constant for every atom (that constant is the
    total-charge multiplier). A mixed mask makes this sensitive to the mask
    itself, not just to a uniformly-applied restraint.
    """
    atoms, grid = _case(seed=8)
    truth = np.array([0.4, -0.3, 0.25, -0.2, -0.15])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)
    mask = np.array([True, False, True, False, True])

    out = resp_fit(
        inv_r,
        v,
        restrain=mask,
        groups=[],
        stage2_free=[],
        total_charge=float(truth.sum()),
    )

    a_matrix, b_vector = normal_equations(inv_r, v)
    q = out.charges
    stationarity = (
        a_matrix @ q - b_vector + np.where(mask, RESP_A_STAGE1 * q / np.sqrt(q**2 + RESP_B**2), 0.0)
    )

    assert stationarity == pytest.approx(np.full(5, stationarity[0]), abs=1e-6)


def test_stage_two_uses_its_own_restraint_strength_not_stage_ones():
    """The existing equivalence test leaves no freedom for a_stage2 to matter:
    every free atom is pinned by an equality row or the total-charge row.
    Here stage2_free has three atoms and no equivalence group, so one degree
    of freedom is genuinely settled by the restrained least squares -- and it
    must move with a_stage2, not sit wherever a_stage1 would have put it.
    """
    atoms, grid = _case(seed=9)
    truth = np.array([-0.4, 0.1, 0.12, 0.09, 0.09])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)
    common = dict(
        restrain=np.ones(5, dtype=bool),
        groups=[],
        stage2_free=[1, 2, 3],
        total_charge=float(truth.sum()),
    )

    correct = resp_fit(inv_r, v, **common)  # a_stage2 defaults to RESP_A_STAGE2
    if_it_used_stage1 = resp_fit(inv_r, v, a_stage2=RESP_A_STAGE1, **common)

    assert not np.allclose(correct.charges[1:4], if_it_used_stage1.charges[1:4], atol=1e-6)
    # atoms outside stage2_free and the total charge are untouched by a_stage2.
    assert correct.charges[0] == pytest.approx(if_it_used_stage1.charges[0], abs=1e-10)
    assert correct.charges[4] == pytest.approx(if_it_used_stage1.charges[4], abs=1e-10)
    assert correct.charges.sum() == pytest.approx(truth.sum(), abs=1e-10)


def test_a_fit_that_will_not_converge_is_refused():
    """Returning an unconverged answer would look like a converged one."""
    atoms, grid = _case(seed=5)
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, np.array([0.4, -0.3, 0.25, -0.2, -0.15]))

    with pytest.raises(ValidationError, match="converge"):
        resp_fit(
            inv_r,
            v,
            restrain=np.ones(5, dtype=bool),
            groups=[],
            stage2_free=[],
            total_charge=0.0,
            max_iterations=1,
            tol=1e-15,
        )


@pytest.mark.needs_orca
def test_unrestrained_resp_on_our_own_grid_reproduces_orca_chelpg(
    butane_pdb, butane_itp, tmp_path, orca_bin
):
    """End-to-end check of the grid, orca_vpot, units and the fit together.

    Does not go through the CLI, because the CLI has no way to switch the
    restraint off. With ``a_stage1 = a_stage2 = 0``, ``resp_fit`` is just a
    constrained least-squares ESP fit -- the same problem ORCA's own CHELPG
    solves, just on a different (MK, not Breneman) grid. If the MK grid, the
    ``orca_vpot`` invocation, the bohr conversion, or the fit itself were
    wrong, this fit would not land anywhere near ORCA's own CHELPG answer;
    agreeing to within 0.06 e is the strongest evidence available that the
    whole RESP wiring in ``cmd_charges`` computes the right numbers, with the
    restraint's own (expected, large) effect on buried atoms excluded from
    the comparison.
    """
    binaries = discover_binaries({"orca": orca_bin})
    topology = read_topology(butane_itp)
    structure = adopt_topology_naming(read_pdb(butane_pdb), topology)
    target_total = derive_charge(topology, None)
    coords_nm = structure.coords

    base = run_esp_job(
        binaries,
        structure,
        coords_nm,
        tmp_path,
        "! HF STO-3G",
        charge=target_total,
        mult=1,
        nprocs=1,
        chelpg=True,
    )

    groups = equivalent_hydrogen_groups(topology)
    raw_chelpg = parse_chelpg_charges(tmp_path / "esp.chelpg.xyz")
    chelpg = average_within_groups(raw_chelpg, groups)

    elements = [element_of(topology, atom.nr) for atom in topology.atoms]
    zero_based = [[nr - 1 for nr in group] for group in groups]
    parents = {
        n - 1
        for group in groups
        for n in topology.bonded_neighbours(group[0])
        if element_of(topology, n) != "H"
    }
    stage2_free = sorted(parents.union(i for g in zero_based for i in g))

    grid_nm = mk_grid(coords_nm, elements, density_per_angstrom2=1.0)
    v_grid = run_vpot(binaries, base, grid_nm, tmp_path)
    inv_r = inverse_distances(nm_to_bohr(coords_nm), nm_to_bohr(grid_nm))

    result = resp_fit(
        inv_r,
        v_grid,
        restrain=np.array([e != "H" for e in elements]),
        groups=zero_based,
        stage2_free=stage2_free,
        total_charge=float(target_total),
        a_stage1=0.0,
        a_stage2=0.0,
    )

    assert np.abs(result.charges - chelpg).max() < 0.06
