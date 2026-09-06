import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.fit import weights as w


def test_uniform_keeps_low_energy_points():
    qm = np.array([0.0, 10.0, 39.0])
    assert w.uniform_weights(qm, emax=40.0) == pytest.approx([1.0, 1.0, 1.0])


def test_uniform_discards_above_emax():
    """Spec section 4.7 default: discard points above 40 kJ/mol."""
    qm = np.array([0.0, 39.0, 41.0, 100.0])
    assert w.uniform_weights(qm, emax=40.0) == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_uniform_is_relative_to_the_minimum():
    qm = np.array([500.0, 510.0, 560.0])
    assert w.uniform_weights(qm, emax=40.0) == pytest.approx([1.0, 1.0, 0.0])


def test_uniform_emax_none_keeps_everything():
    qm = np.array([0.0, 1000.0])
    assert w.uniform_weights(qm, emax=None) == pytest.approx([1.0, 1.0])


def test_boltzmann_decreases_with_energy():
    qm = np.array([0.0, 5.0, 20.0])
    out = w.boltzmann_weights(qm, temperature=300.0, emax=None)
    assert out[0] > out[1] > out[2]
    assert out[0] == pytest.approx(1.0)


def test_boltzmann_matches_formula():
    qm = np.array([0.0, 5.0])
    out = w.boltzmann_weights(qm, temperature=300.0, emax=None)
    expected = np.exp(-5.0 / (w.GAS_CONSTANT_KJ * 300.0))
    assert out[1] == pytest.approx(expected)


def test_boltzmann_still_applies_emax_cut():
    qm = np.array([0.0, 100.0])
    out = w.boltzmann_weights(qm, temperature=300.0, emax=40.0)
    assert out[1] == 0.0


def test_all_weights_non_negative():
    qm = np.linspace(0, 80, 17)
    assert np.all(w.uniform_weights(qm) >= 0)
    assert np.all(w.boltzmann_weights(qm) >= 0)


def test_boltzmann_rejects_zero_temperature():
    """Temperature <= 0 must raise ValidationError to prevent NaN output."""
    qm = np.array([0.0, 5.0, 20.0])
    with pytest.raises(ValidationError, match="temperature must be > 0"):
        w.boltzmann_weights(qm, temperature=0.0, emax=None)
    with pytest.raises(ValidationError, match="temperature must be > 0"):
        w.boltzmann_weights(qm, temperature=-5.0, emax=None)
    # Valid temperature still works
    result = w.boltzmann_weights(qm, temperature=300.0, emax=None)
    assert result[0] == pytest.approx(1.0)
