"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, Momentum-Rausch-Anfaelligkeit
(Hook 1, verfeinert), Adams (Nicht-)Vorteil unter Rauschen (Hook 2, widerlegt bei dieser
Dimension), rauschfreie Baseline (Hook 3), exakte Reduktions-/Bias-Korrektur-Checks (Hook 4+5),
Gradienten-Check."""
from dataclasses import dataclass

import numpy as np

import sgd_constants as C
import sgd_functions as fn
import sgd_optimizers as opt


@dataclass(frozen=True)
class Settings:
    sigma: float


def analyse(settings: Settings) -> dict:
    trajectories = {}
    for name, run_fn, kwargs in _display_runs(settings.sigma):
        res = run_fn(seed=0, n_iter=C.N_ITER_DISPLAY, tol=C.TOL_SUCCESS,
                     record_every=C.RECORD_EVERY_DISPLAY, **kwargs)
        trajectories[name] = res
    return {"trajectories": trajectories}


def _display_runs(sigma):
    return [
        ("SGD", opt.sgd_run, dict(eta=C.ETA_SGD, sigma=sigma)),
        ("Momentum", opt.momentum_run, dict(eta=C.ETA_SGD, beta=C.BETA_MOMENTUM, sigma=sigma)),
        ("Adam", opt.adam_run, dict(eta=C.ETA_ADAM, beta1=C.ADAM_BETA1, beta2=C.ADAM_BETA2,
                                    eps=C.ADAM_EPS, sigma=sigma)),
    ]


def noise_free_baseline_check() -> list:
    """Bei sigma=0 laufen alle drei Verfahren zum selben, aus Stueck 1 bekannten Rosenbrock-
    Minimum (1,1) - innerhalb des Iterationsbudgets liegen SGD und Adam noch spuerbar naeher am
    Rand der Toleranz als Momentum (das hier klassisch beschleunigt, siehe Hook 1)."""
    rows = []
    for name, run_fn, kwargs in _display_runs(0.0):
        res = run_fn(seed=0, n_iter=C.N_ITER_SWEEP, tol=C.TOL_SUCCESS, **kwargs)
        err = float(np.linalg.norm(res.x_final - fn.X_STAR))
        rows.append({"optimizer": name, "err": err, "within_tolerance": err < C.TOL_SUCCESS})
    return rows


def momentum_replication_check(sigmas=(0.0, 2.0, 5.0, 10.0, 20.0, 50.0),
                               n_seeds: int = C.N_SEEDS_SWEEP) -> list:
    """Hook 1, verfeinert: bei GLEICHER Lernrate ist Momentum im RAUSCHFREIEN Fall nicht
    schlechter als SGD (eher besser, klassische Beschleunigung) - aber die Erfolgsquote bricht
    mit wachsendem Rauschen deutlich staerker ein als bei SGD."""
    rows = []
    for sigma in sigmas:
        sgd = opt.success_rate(opt.sgd_run, n_seeds=n_seeds, eta=C.ETA_SGD, sigma=sigma,
                               n_iter=C.N_ITER_SWEEP, tol=C.TOL_SUCCESS)
        mom = opt.success_rate(opt.momentum_run, n_seeds=n_seeds, eta=C.ETA_SGD,
                               beta=C.BETA_MOMENTUM, sigma=sigma, n_iter=C.N_ITER_SWEEP,
                               tol=C.TOL_SUCCESS)
        rows.append({"sigma": sigma, "sgd_successes": sgd["successes"],
                    "momentum_successes": mom["successes"], "n_seeds": n_seeds,
                    "momentum_worse": mom["successes"] < sgd["successes"]})
    return rows


def noise_sweep_check(sigmas=(0.0, 2.0, 5.0, 10.0, 20.0, 50.0),
                      n_seeds: int = C.N_SEEDS_SWEEP) -> list:
    """Hook 2: waechst Adams relativer Vorsprung gegenueber SGD mit dem Rauschen? Gemessen fuer
    alle drei Verfahren gemeinsam (zentrale Messreihe der App)."""
    rows = []
    for sigma in sigmas:
        sgd = opt.success_rate(opt.sgd_run, n_seeds=n_seeds, eta=C.ETA_SGD, sigma=sigma,
                               n_iter=C.N_ITER_SWEEP, tol=C.TOL_SUCCESS)
        mom = opt.success_rate(opt.momentum_run, n_seeds=n_seeds, eta=C.ETA_SGD,
                               beta=C.BETA_MOMENTUM, sigma=sigma, n_iter=C.N_ITER_SWEEP,
                               tol=C.TOL_SUCCESS)
        adam = opt.success_rate(opt.adam_run, n_seeds=n_seeds, eta=C.ETA_ADAM,
                                beta1=C.ADAM_BETA1, beta2=C.ADAM_BETA2, eps=C.ADAM_EPS,
                                sigma=sigma, n_iter=C.N_ITER_SWEEP, tol=C.TOL_SUCCESS)
        rows.append({"sigma": sigma, "sgd": sgd["successes"], "momentum": mom["successes"],
                    "adam": adam["successes"], "n_seeds": n_seeds})
    return rows


def adam_advantage_anisotropic_check(sigma_x: float = 2.0, sigma_y: float = 40.0,
                                     n_seeds: int = C.N_SEEDS_SWEEP) -> dict:
    """Robustheits-Check zu Hook 2: selbst unter STARK heterogenem Rauschen zwischen den beiden
    Dimensionen (Adams klassische Staerke) zeigt sich bei nur zwei Parametern kein Vorteil
    gegenueber einer gut getunten einzelnen Schrittweite fuer SGD - anders als im viel
    hochdimensionaleren Gewichtsraum von mlp-backprop-demo."""

    def aniso_grad(x, rng):
        g = fn.grad_rosenbrock(x)
        return g + np.array([sigma_x * rng.standard_normal(), sigma_y * rng.standard_normal()])

    def run_sgd(eta, seed, n_iter):
        rng = np.random.default_rng(seed)
        x = opt.ROSENBROCK_X0.copy()
        for _ in range(n_iter):
            x = x - eta * aniso_grad(x, rng)
            if not np.all(np.isfinite(x)):
                return x
        return x

    def run_adam(eta, seed, n_iter):
        rng = np.random.default_rng(seed)
        x = opt.ROSENBROCK_X0.copy()
        m = np.zeros(2)
        v = np.zeros(2)
        for k in range(1, n_iter + 1):
            g = aniso_grad(x, rng)
            m = C.ADAM_BETA1 * m + (1 - C.ADAM_BETA1) * g
            v = C.ADAM_BETA2 * v + (1 - C.ADAM_BETA2) * g ** 2
            m_hat = m / (1 - C.ADAM_BETA1 ** k)
            v_hat = v / (1 - C.ADAM_BETA2 ** k)
            x = x - eta * m_hat / (np.sqrt(v_hat) + C.ADAM_EPS)
            if not np.all(np.isfinite(x)):
                return x
        return x

    def best_success(run_fn, etas):
        best = 0
        for eta in etas:
            succ = sum(
                1 for seed in range(n_seeds)
                if np.linalg.norm(run_fn(eta, seed, C.N_ITER_SWEEP) - fn.X_STAR) < C.TOL_SUCCESS
            )
            best = max(best, succ)
        return best

    sgd_best = best_success(run_sgd, [0.0005, 0.001, 0.0015, 0.002])
    adam_best = best_success(run_adam, [0.01, 0.02, 0.03, 0.05])
    return {"sigma_x": sigma_x, "sigma_y": sigma_y, "sgd_best": sgd_best, "adam_best": adam_best,
            "n_seeds": n_seeds, "adam_advantage": adam_best > sgd_best}


def momentum_reduces_to_sgd_check(sigma: float = 3.0, n_iter: int = 50) -> dict:
    """Hook 4: Momentum mit beta=0 reduziert sich algebraisch EXAKT auf SGD."""
    sgd_res = opt.sgd_run(eta=0.001, sigma=sigma, seed=0, n_iter=n_iter, tol=C.TOL_SUCCESS)
    mom_res = opt.momentum_run(eta=0.001, beta=0.0, sigma=sigma, seed=0, n_iter=n_iter,
                               tol=C.TOL_SUCCESS)
    max_abs_err = float(np.max(np.abs(sgd_res.x_final - mom_res.x_final)))
    return {"max_abs_err": max_abs_err}


def adam_bias_correction_check(sigma: float = 3.0) -> dict:
    """Hook 5: Adams bias-korrigiertes erstes Moment ist im allerersten Schritt exakt gleich dem
    (verrauschten) Gradienten selbst, unabhaengig von beta1."""
    rng = np.random.default_rng(1)
    g1 = fn.noisy_gradient(opt.ROSENBROCK_X0, sigma, rng)
    errs = {}
    for beta1 in (0.5, 0.9, 0.99):
        m1 = (1 - beta1) * g1
        m1_hat = m1 / (1 - beta1 ** 1)
        errs[beta1] = float(np.max(np.abs(m1_hat - g1)))
    return {"errs": errs, "max_err": max(errs.values())}


def gradient_check(eps: float = 1e-6) -> dict:
    x = np.array([0.3, -0.4])
    analytic = fn.grad_rosenbrock(x)
    numeric = np.zeros(2)
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (fn.rosenbrock(xp) - fn.rosenbrock(xm)) / (2 * eps)
    return {"max_rel_err": float(np.max(np.abs(analytic - numeric) /
                                        np.maximum(np.abs(analytic), 1e-8)))}
