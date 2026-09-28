"""Drei stochastische Gradientenverfahren, alle mit dem GLEICHEN verrauschten Gradienten
(sgd_functions.noisy_gradient), eigene numpy-Implementierungen wie die ganze Linie.
ROSENBROCK_X0 = (-1,2; 1,0) - derselbe klassische Startpunkt wie in Stueck 3 (kein Regler,
Konvention der Linie)."""
from dataclasses import dataclass

import numpy as np

import sgd_functions as fn

ROSENBROCK_X0 = np.array([-1.2, 1.0])


@dataclass
class Result:
    trajectory: list
    x_final: np.ndarray
    finite: bool
    success: bool


def _finish(x, trajectory, tol):
    finite = bool(np.all(np.isfinite(x)))
    success = finite and float(np.linalg.norm(x - fn.X_STAR)) < tol
    return Result(trajectory=trajectory, x_final=x, finite=finite, success=success)


def sgd_run(eta: float, sigma: float, seed: int, n_iter: int, tol: float = 0.5,
           x0=None, record_every: int = 1) -> Result:
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float) if x0 is not None else ROSENBROCK_X0.copy()
    trajectory = [(float(x[0]), float(x[1]))]
    for k in range(n_iter):
        g = fn.noisy_gradient(x, sigma, rng)
        x = x - eta * g
        if not np.all(np.isfinite(x)):
            return _finish(x, trajectory, tol)
        if (k + 1) % record_every == 0:
            trajectory.append((float(x[0]), float(x[1])))
    return _finish(x, trajectory, tol)


def momentum_run(eta: float, beta: float, sigma: float, seed: int, n_iter: int, tol: float = 0.5,
                 x0=None, record_every: int = 1) -> Result:
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float) if x0 is not None else ROSENBROCK_X0.copy()
    v = np.zeros(2)
    trajectory = [(float(x[0]), float(x[1]))]
    for k in range(n_iter):
        g = fn.noisy_gradient(x, sigma, rng)
        v = beta * v - eta * g
        x = x + v
        if not np.all(np.isfinite(x)):
            return _finish(x, trajectory, tol)
        if (k + 1) % record_every == 0:
            trajectory.append((float(x[0]), float(x[1])))
    return _finish(x, trajectory, tol)


def adam_run(eta: float, beta1: float, beta2: float, eps: float, sigma: float, seed: int,
            n_iter: int, tol: float = 0.5, x0=None, record_every: int = 1) -> Result:
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float) if x0 is not None else ROSENBROCK_X0.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    trajectory = [(float(x[0]), float(x[1]))]
    for k in range(1, n_iter + 1):
        g = fn.noisy_gradient(x, sigma, rng)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** k)
        v_hat = v / (1 - beta2 ** k)
        x = x - eta * m_hat / (np.sqrt(v_hat) + eps)
        if not np.all(np.isfinite(x)):
            return _finish(x, trajectory, tol)
        if k % record_every == 0:
            trajectory.append((float(x[0]), float(x[1])))
    return _finish(x, trajectory, tol)


def success_rate(run_fn, n_seeds: int = 30, **kwargs) -> dict:
    successes = 0
    for seed in range(n_seeds):
        res = run_fn(seed=seed, **kwargs)
        if res.success:
            successes += 1
    return {"successes": successes, "n_seeds": n_seeds}
