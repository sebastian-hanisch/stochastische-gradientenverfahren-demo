"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen neu berechnet."""
import pytest

import sgd_evaluation as ev


@pytest.fixture(scope="module")
def noise_sweep_rows():
    return ev.noise_sweep_check()


def test_claim_all_optimizers_converge_without_noise():
    rows = ev.noise_free_baseline_check()
    assert all(row["within_tolerance"] for row in rows)


def test_claim_momentum_not_worse_without_noise(noise_sweep_rows):
    zero_row = [r for r in noise_sweep_rows if r["sigma"] == 0.0][0]
    assert zero_row["momentum"] >= zero_row["sgd"]


def test_claim_momentum_collapses_under_strong_noise(noise_sweep_rows):
    """Ehrlicher, verfeinerter Befund: Momentum bricht unter Rauschen deutlich staerker ein als
    SGD - plattformrobust als 'klar weniger Erfolge' geprueft, nicht auf einen exakten Zaehlwert
    gepinnt (siehe feedback_ci_platform_robust_tests)."""
    strong_row = [r for r in noise_sweep_rows if r["sigma"] == 20.0][0]
    assert strong_row["momentum"] < strong_row["sgd"] - 5


def test_claim_adam_shows_no_clear_advantage_over_sgd(noise_sweep_rows):
    """Ehrlicher Befund: anders als bei mlp-backprop-demo zeigt Adam hier keinen klaren Vorsprung
    - Erfolgszahlen bleiben nah an SGD ueber alle Rauschstaerken."""
    for row in noise_sweep_rows:
        assert abs(row["adam"] - row["sgd"]) <= 5


def test_claim_momentum_reduces_to_sgd_exactly():
    out = ev.momentum_reduces_to_sgd_check()
    assert out["max_abs_err"] < 1e-10


def test_claim_adam_bias_correction_exact_at_first_step():
    out = ev.adam_bias_correction_check()
    assert out["max_err"] < 1e-10


def test_claim_gradient_check_below_1e_minus_6():
    out = ev.gradient_check()
    assert out["max_rel_err"] < 1e-6
