"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

SIGMA_MIN, SIGMA_MAX, SIGMA_DEFAULT = 0.0, 50.0, 10.0
ETA_SGD = 0.0015
ETA_ADAM = 0.02
BETA_MOMENTUM = 0.9
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
N_ITER_SWEEP = 3000
N_SEEDS_SWEEP = 30
TOL_SUCCESS = 0.5
N_ITER_DISPLAY = 500
RECORD_EVERY_DISPLAY = 5

PRESETS = {
    "rauschfrei": dict(
        label="Rauschfrei",
        sigma=0.0,
        help="Ohne Rauschen konvergieren alle drei Verfahren zum selben Minimum — Momentum sogar "
             "schneller als SGD (klassisches Beschleunigungsverhalten).",
    ),
    "starkes_rauschen": dict(
        label="Starkes Rauschen",
        sigma=20.0,
        help="Bei starkem Gradientenrauschen bricht Momentum praktisch vollständig ein — SGD und "
             "Adam bleiben etwa gleich robust.",
    ),
}
