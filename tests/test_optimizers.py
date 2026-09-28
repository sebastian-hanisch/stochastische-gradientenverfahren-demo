import numpy as np

import sgd_optimizers as opt


def test_sgd_converges_close_without_noise():
    res = opt.sgd_run(eta=0.002, sigma=0.0, seed=0, n_iter=3000)
    assert res.finite
    assert np.linalg.norm(res.x_final - np.array([1.0, 1.0])) < 0.2


def test_momentum_reduces_to_sgd_when_beta_zero():
    sgd_res = opt.sgd_run(eta=0.001, sigma=3.0, seed=0, n_iter=50)
    mom_res = opt.momentum_run(eta=0.001, beta=0.0, sigma=3.0, seed=0, n_iter=50)
    assert np.max(np.abs(sgd_res.x_final - mom_res.x_final)) < 1e-12


def test_momentum_converges_without_noise():
    res = opt.momentum_run(eta=0.002, beta=0.9, sigma=0.0, seed=0, n_iter=3000)
    assert res.finite
    assert np.linalg.norm(res.x_final - np.array([1.0, 1.0])) < 1e-4


def test_adam_converges_without_noise():
    res = opt.adam_run(eta=0.02, beta1=0.9, beta2=0.999, eps=1e-8, sigma=0.0, seed=0,
                       n_iter=3000)
    assert res.finite
    assert np.linalg.norm(res.x_final - np.array([1.0, 1.0])) < 0.1


def test_success_rate_counts_correctly():
    out = opt.success_rate(opt.sgd_run, n_seeds=5, eta=0.002, sigma=0.0, n_iter=3000)
    assert out["n_seeds"] == 5
    assert 0 <= out["successes"] <= 5


def test_trajectory_records_expected_points():
    res = opt.sgd_run(eta=0.001, sigma=0.0, seed=0, n_iter=20, record_every=5)
    assert len(res.trajectory) == 1 + 4
