import numpy as np
import pytest
from ffrefine.fit.selection import (
    Candidate,
    bic,
    format_ranking_table,
    rank_candidates,
    select_model,
)
from ffrefine.mm.analytic import series_energy


@pytest.fixture
def threefold():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.5]), np.array([0.0]), np.array([3]), phi)
    return phi, target


def test_bic_penalises_more_parameters():
    assert bic(1.0, 100, 5) > bic(1.0, 100, 2)


def test_bic_rewards_lower_error():
    assert bic(0.5, 100, 3) < bic(2.0, 100, 3)


def test_rank_enumerates_all_subsets(threefold):
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3])
    assert len(cands) == 7  # 2^3 - 1


def test_rank_enumerates_31_for_default_multiplicities(threefold):
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    assert len(cands) == 31


def test_rank_sorted_by_bic_ascending(threefold):
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    assert all(a.bic <= b.bic for a, b in zip(cands, cands[1:]))


def test_selects_minimal_model_for_pure_threefold(threefold):
    """A pure n=3 profile must select exactly {3}, not a larger set."""
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    chosen = select_model(cands)
    assert chosen.mults == (3,)


def test_selects_both_terms_for_two_term_profile():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0, 1.0]), np.array([0.0, 180.0]), np.array([3, 1]), phi)
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    chosen = select_model(cands)
    assert set(chosen.mults) == {1, 3}


def test_select_prefers_fewer_terms_within_tolerance():
    """Given two models within the RMSE tolerance, the smaller wins."""
    small = Candidate(mults=(3,), result=None, bic=10.0, rmse=0.40, n_params=2)
    big = Candidate(mults=(1, 2, 3, 4), result=None, bic=9.0, rmse=0.10, n_params=5)
    assert select_model([big, small], rmse_tolerance=0.5).mults == (3,)


def test_select_rejects_too_poor_a_model():
    small = Candidate(mults=(3,), result=None, bic=10.0, rmse=5.0, n_params=2)
    big = Candidate(mults=(1, 2, 3), result=None, bic=9.0, rmse=0.10, n_params=4)
    assert select_model([big, small], rmse_tolerance=0.5).mults == (1, 2, 3)


def test_ranking_table_is_printable(threefold):
    phi, target = threefold
    text = format_ranking_table(rank_candidates(phi, target, [1, 2, 3]))
    assert "BIC" in text and "RMSE" in text
    assert "3" in text
