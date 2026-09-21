import numpy as np
import pytest

from ffrefine.charges.esp import inverse_distances, potential_from_charges
from ffrefine.charges.resp import resp_fit
from ffrefine.errors import ValidationError


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
