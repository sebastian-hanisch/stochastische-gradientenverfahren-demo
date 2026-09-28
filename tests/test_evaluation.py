import pytest

import sgd_evaluation as ev


@pytest.fixture(scope="module")
def noise_sweep_rows():
    return ev.noise_sweep_check(sigmas=(0.0, 10.0, 20.0), n_seeds=15)


@pytest.fixture(scope="module")
def momentum_rows():
    return ev.momentum_replication_check(sigmas=(0.0, 10.0, 20.0), n_seeds=15)


def test_analyse_returns_three_trajectories():
    out = ev.analyse(ev.Settings(sigma=5.0))
    assert set(out["trajectories"].keys()) == {"SGD", "Momentum", "Adam"}


def test_noise_free_baseline_all_within_tolerance():
    rows = ev.noise_free_baseline_check()
    assert all(row["within_tolerance"] for row in rows)


def test_momentum_not_worse_without_noise(momentum_rows):
    zero_row = [r for r in momentum_rows if r["sigma"] == 0.0][0]
    assert not zero_row["momentum_worse"]


def test_momentum_worse_under_strong_noise(momentum_rows):
    strong_row = [r for r in momentum_rows if r["sigma"] == 20.0][0]
    assert strong_row["momentum_worse"]


def test_noise_sweep_momentum_degrades_monotonically(noise_sweep_rows):
    momentum_vals = [r["momentum"] for r in noise_sweep_rows]
    assert all(b <= a for a, b in zip(momentum_vals, momentum_vals[1:]))


def test_momentum_reduces_to_sgd_check():
    out = ev.momentum_reduces_to_sgd_check()
    assert out["max_abs_err"] < 1e-10


def test_adam_bias_correction_check():
    out = ev.adam_bias_correction_check()
    assert out["max_err"] < 1e-10


def test_adam_advantage_anisotropic_check_runs():
    out = ev.adam_advantage_anisotropic_check(n_seeds=10)
    assert 0 <= out["sgd_best"] <= 10
    assert 0 <= out["adam_best"] <= 10


def test_gradient_check_below_threshold():
    out = ev.gradient_check()
    assert out["max_rel_err"] < 1e-6
