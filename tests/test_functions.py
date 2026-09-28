import numpy as np

import sgd_functions as fn


def test_rosenbrock_minimum_is_zero_at_star():
    assert fn.rosenbrock(fn.X_STAR) == 0.0


def test_gradient_matches_finite_differences():
    eps = 1e-6
    x = np.array([0.3, -0.4])
    analytic = fn.grad_rosenbrock(x)
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric = (fn.rosenbrock(xp) - fn.rosenbrock(xm)) / (2 * eps)
        assert abs(analytic[i] - numeric) < 1e-3


def test_noisy_gradient_matches_true_gradient_at_zero_sigma():
    rng = np.random.default_rng(0)
    x = np.array([0.2, 0.5])
    assert np.array_equal(fn.noisy_gradient(x, 0.0, rng), fn.grad_rosenbrock(x))


def test_noisy_gradient_is_reproducible_with_fixed_seed():
    x = np.array([0.2, 0.5])
    g1 = fn.noisy_gradient(x, 3.0, np.random.default_rng(42))
    g2 = fn.noisy_gradient(x, 3.0, np.random.default_rng(42))
    assert np.array_equal(g1, g2)
