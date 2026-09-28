"""Vehikel C: eine verrauschte Variante von Vehikel A (Rosenbrock-Funktion, eigene Kopie aus
Stueck 1-3, kein Import). Bei jedem Schritt wird dem exakten Gradienten gaussches Rauschen
zugefuegt - das klassische synthetische Testbed fuer stochastische Gradientenverfahren (Robbins &
Monro 1951), ohne echten Datensatz oder neuronales Netz."""
import numpy as np

A, B = 1.0, 100.0
X_STAR = np.array([1.0, 1.0])


def rosenbrock(x: np.ndarray) -> float:
    return (A - x[0]) ** 2 + B * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
    dx = -2 * (A - x[0]) - 4 * B * x[0] * (x[1] - x[0] ** 2)
    dy = 2 * B * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


def noisy_gradient(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """g_noisy(x) = grad f(x) + sigma * N(0, I) - simuliert Mini-Batch-/Stichproben-Rauschen."""
    return grad_rosenbrock(x) + sigma * rng.standard_normal(2)
