"""Unabhaengige Orakel-Tests (anderer Rechenweg als der Code):
- Rosenbrock-Funktion/Gradient gegen scipy.optimize.rosen/rosen_der.
- Referenz-Laeufe: alle Seeds gleichzeitig als numpy-Arrays (vektorisiert), das Rauschen vorab als
  (n, 2)-Block gezogen (derselbe Generator-Strom wie die Einzelziehungen im Code); Momentum in
  der Zwei-Punkt-Form x+ = x - eta*g + beta*(x - x_alt) statt der Geschwindigkeitsform; Adam mit
  ausgeschriebenen Momenten. Pfade (kurze Laeufe) und Erfolgszaehler (10 Seeds, 3000 Schritte).
- Adam, erster Schritt: x1 = x0 - eta*g/(|g|+eps) (bias-korrigiert), direkt am Optimierer geprueft.
Die Erfolgszaehler duerfen um 1 abweichen: Seeds nahe der Toleranzgrenze kippen schon durch
algebraisch aequivalente Rundung (plattformabhaengig, siehe README)."""
import numpy as np
import pytest

import sgd_constants as C
import sgd_functions as fn
import sgd_optimizers as opt

scipy_opt = pytest.importorskip("scipy.optimize")
X0 = np.array([-1.2, 1.0])


def _ref(seeds, sigma, n_iter, kind, eta, beta=0.9, b1=0.9, b2=0.999, eps=1e-8):
    noise = np.stack([np.random.default_rng(s).standard_normal((n_iter, 2)) for s in seeds])
    X = np.tile(X0, (len(seeds), 1))
    X_old = X.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    alive = np.ones(len(seeds), bool)
    with np.errstate(all="ignore"):
        for k in range(1, n_iter + 1):
            x, y = X[:, 0], X[:, 1]
            g = np.stack([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)], 1)
            g = g + sigma * noise[:, k - 1]
            if kind == "sgd":
                Xn = X - eta * g
            elif kind == "mom":
                Xn = X - eta * g + beta * (X - X_old)
                X_old = X
            else:
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                Xn = X - eta * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + eps)
            X = np.where(alive[:, None], Xn, X)
            alive &= np.all(np.isfinite(X), 1)
    return X, alive


def test_rosenbrock_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.normal(size=2) * 2
        assert fn.rosenbrock(x) == pytest.approx(scipy_opt.rosen(x))
        np.testing.assert_allclose(fn.grad_rosenbrock(x), scipy_opt.rosen_der(x))


def test_short_runs_match_vectorised_reference():
    rng = np.random.default_rng(3)
    for t in range(45):
        kind = ("sgd", "mom", "adam")[t % 3]
        sigma = float(rng.choice([0, 1, 3, 10]))
        seed, n_iter = int(rng.integers(0, 1000)), int(rng.integers(20, 300))
        if kind == "adam":
            eta = float(rng.uniform(0.005, 0.05))
            r = opt.adam_run(eta, 0.9, 0.999, 1e-8, sigma, seed, n_iter)
            X, alive = _ref([seed], sigma, n_iter, kind, eta)
        else:
            eta = float(rng.uniform(0.0005, 0.003))
            beta = float(rng.choice([0.0, 0.5, 0.9]))
            r = (opt.sgd_run(eta, sigma, seed, n_iter) if kind == "sgd"
                 else opt.momentum_run(eta, beta, sigma, seed, n_iter))
            X, alive = _ref([seed], sigma, n_iter, kind, eta, beta=beta)
        assert r.finite == bool(alive[0])
        assert len(r.trajectory) == n_iter + 1
        np.testing.assert_allclose(r.x_final, X[0], rtol=1e-6, atol=1e-8)
        assert r.success == bool(np.linalg.norm(X[0] - 1) < C.TOL_SUCCESS)


def test_adam_first_step_is_signed_learning_rate():
    rng = np.random.default_rng(4)
    for _ in range(30):
        sigma, seed, eta = float(rng.uniform(0, 30)), int(rng.integers(0, 999)), float(rng.uniform(0.005, 0.05))
        r = opt.adam_run(eta, 0.9, 0.999, 1e-8, sigma, seed, 1)
        g = fn.grad_rosenbrock(X0) + sigma * np.random.default_rng(seed).standard_normal(2)
        np.testing.assert_allclose(r.x_final, X0 - eta * g / (np.abs(g) + 1e-8), rtol=1e-9)


def test_success_counts_match_reference_over_10_seeds():
    seeds = list(range(10))
    for sigma in (5.0, 20.0):
        for kind, eta, run in (
            ("sgd", C.ETA_SGD, lambda s: opt.sgd_run(C.ETA_SGD, sigma, s, C.N_ITER_SWEEP, C.TOL_SUCCESS)),
            ("adam", C.ETA_ADAM, lambda s: opt.adam_run(C.ETA_ADAM, C.ADAM_BETA1, C.ADAM_BETA2,
                                                         C.ADAM_EPS, sigma, s, C.N_ITER_SWEEP, C.TOL_SUCCESS)),
        ):
            X, alive = _ref(seeds, sigma, C.N_ITER_SWEEP, kind, eta)
            ref = int((alive & (np.linalg.norm(X - 1, axis=1) < C.TOL_SUCCESS)).sum())
            demo = sum(run(s).success for s in seeds)
            assert abs(demo - ref) <= 1, (kind, sigma, demo, ref)
