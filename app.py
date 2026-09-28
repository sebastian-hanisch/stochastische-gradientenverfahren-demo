"""Stochastische Gradientenverfahren — die Brücke zu Deep Learning

Sebastian Hanisch - Operations Research und Machine Learning

Stück 8 (LETZTES) der "Nichtlineare Optimierung"-Reihe der "Konzepte"-Reihe:
Gradientenabstieg -> Newton-Verfahren -> Quasi-Newton -> Lagrange/KKT ->
{Straf-/Barriere-Verfahren, SQP -> Innere-Punkte-Verfahren} + Stochastische Gradientenverfahren.
Stück 2-7 behoben fehlende Krümmungsinformation (Newton-Ast). Dieses Stück behebt eine ANDERE
Ignoranz direkt von der Wurzel aus: teure/verrauschte Gradienten bei großem n - genau die
Situation beim Training eines neuronalen Netzes (Mini-Batch statt vollem Datensatz).

Lauffähig mit: streamlit run app.py
"""
import streamlit as st

import sgd_constants as C
import sgd_evaluation as ev
import sgd_presets as pr
import sgd_visualization as viz

st.set_page_config(page_title="Stochastische Gradientenverfahren", layout="wide")


@st.cache_data(show_spinner=False)
def _run(sigma):
    out = ev.analyse(ev.Settings(sigma=sigma))
    return out["trajectories"]


@st.cache_data(show_spinner="Rausch-Sweep läuft (30 Seeds × 3 Verfahren × 6 Rauschstärken)…")
def _noise_sweep():
    return ev.noise_sweep_check()


@st.cache_data(show_spinner=False)
def _baseline():
    return ev.noise_free_baseline_check()


@st.cache_data(show_spinner=False)
def _reduces_to_sgd():
    return ev.momentum_reduces_to_sgd_check()


@st.cache_data(show_spinner=False)
def _bias_correction():
    return ev.adam_bias_correction_check()


@st.cache_data(show_spinner="Anisotropes Rauschen wird geprüft…")
def _aniso():
    return ev.adam_advantage_anisotropic_check()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


st.title("🌊 Stochastische Gradientenverfahren")
st.markdown(
    "Stück 2–7 behoben fehlende Krümmungsinformation. Dieses Stück behebt eine ANDERE Ignoranz "
    "direkt von der Wurzel (Gradientenabstieg) aus: **teure/verrauschte Gradienten** bei großem "
    "$n$ — genau die Situation beim Training eines neuronalen Netzes (Mini-Batch statt vollem "
    "Datensatz, Robbins & Monro 1951). Vehikel: dieselbe Rosenbrock-Funktion aus Stück 1–3, jetzt "
    "mit gaußschem Rauschen auf dem Gradienten — **kein neuronales Netz**, damit der Optimierer "
    "selbst im Zentrum steht."
)
st.caption(
    "Stück 8 (LETZTES) der 'Nichtlineare Optimierung'-Reihe. Vergleich mit "
    "[mlp-backprop-demo](https://github.com/sebastian-hanisch/mlp-backprop-demo) "
    "(Stück 2 der Neuronale-Netze-Reihe): dieselben drei Verfahren, dort auf einem echten "
    "2-Schichten-Netz auf XOR."
)

with st.expander("So funktionieren die drei Verfahren", expanded=True):
    st.markdown(
        "1. **SGD** (Robbins & Monro 1951): $x_{k+1}=x_k-\\eta\\,g_{\\text{noisy}}(x_k)$, mit "
        "$g_{\\text{noisy}}=\\nabla f+\\sigma\\varepsilon$.\n"
        "2. **Momentum/Heavy-Ball** (Polyak 1964): $v_{k+1}=\\beta v_k-\\eta\\,g_{\\text{noisy}}"
        "(x_k)$, $x_{k+1}=x_k+v_{k+1}$ — akkumuliert Geschwindigkeit, beschleunigt im "
        "rauschfreien Fall, verstärkt aber auch korreliertes Rauschen.\n"
        "3. **Adam** (Kingma & Ba 2015): bias-korrigierte erste/zweite Momente, passt die "
        "Schrittweite pro Parameter an."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                  use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    sigma = st.slider("Rauschstärke σ", C.SIGMA_MIN, C.SIGMA_MAX, ss["sigma"], step=1.0,
                      key="widget_sigma", on_change=pr.store_from_widget, args=("sigma",))
    ss["sigma"] = sigma
    st.caption(f"Feste Lernraten: SGD/Momentum η={C.ETA_SGD}, Adam η={C.ETA_ADAM}, "
              f"β_Momentum={C.BETA_MOMENTUM}")

pr.sync_query_params(dict(sigma=sigma))

trajectories = _run(sigma)

st.markdown("---")
st.subheader("🎯 Eine Trajektorie (illustrativ, ein Seed)")
col_left, col_right = st.columns([3, 2])
with col_left:
    fig = viz.build_trajectory_figure(trajectories, title=f"σ={sigma:.0f}")
    st.plotly_chart(fig, key=f"traj_{sigma}", use_container_width=True)
with col_right:
    for name, res in trajectories.items():
        st.metric(f"{name}: Endpunkt", f"({res.x_final[0]:.3f}, {res.x_final[1]:.3f})")
    st.caption(
        "Ein einzelner Seed dient nur der Anschauung — die zentrale Messung unten mittelt über "
        f"{C.N_SEEDS_SWEEP} Seeds."
    )

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: Erfolgsquote vs. Rauschstärke")
sweep = _noise_sweep()
st.plotly_chart(viz.build_noise_sweep_figure(sweep), key="sweep", use_container_width=True)
st.caption(
    "**Verfeinerter Befund (nicht die naive Erwartung):** bei GLEICHER Lernrate ist Momentum im "
    "rauschfreien Fall NICHT schlechter als SGD — im Gegenteil, es konvergiert schneller "
    "(klassische Beschleunigung, siehe 📐). Erst mit wachsendem Gradientenrauschen bricht seine "
    "Erfolgsquote deutlich stärker ein als bei SGD, weil die akkumulierte Geschwindigkeit "
    "korreliertes Rauschen verstärkt. **Adam zeigt hier KEINEN klaren Vorteil gegenüber SGD** — "
    "anders als bei `mlp-backprop-demo` (9/10 gegen 7/10 auf XOR): bei nur zwei Parametern "
    "gleicht eine gut getunte einzelne Schrittweite Adams Pro-Parameter-Anpassung fast "
    "vollständig aus (siehe 📐)."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Momentum ist immer besser als SGD | Gilt nur rauschfrei — unter Rauschen kehrt sich das "
    "um (siehe 🎯 oben) | Adaptive Verfahren wie Adam dämpfen das teilweise |\n"
    "| Adam ist immer besser als SGD | Gilt hier NICHT bei nur 2 Parametern, auch nicht unter "
    "heterogenem Rauschen (siehe 📐) — bei mlp-backprop-demos vielen Gewichten schon | Adams "
    "Vorteil ist dimensionsabhängig |\n"
    "| Muon schlägt Adam | Auf diesem 2-Parameter-Vehikel kein stabiler, echter Unterschied "
    "gemessen (siehe 📐) — bleibt reiner Literatur-Ausblick, nicht gebaut | Erst bei sehr hoher "
    "Dimension (LLM-Pretraining) zeigt sich der Effekt laut Literatur |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
$g_{\text{noisy}}(x)=\nabla f(x)+\sigma\varepsilon,\ \varepsilon\sim\mathcal N(0,I)$, angewendet
auf die Rosenbrock-Funktion aus Stück 1–3.
"""
    )
    st.markdown("**Rauschfreie Baseline** (σ=0, 3000 Iterationen, ein Seed):")
    baseline = _baseline()
    for row in baseline:
        st.caption(f"{row['optimizer']}: Fehler zum Minimum = {row['err']:.2e}")

    st.markdown("**Exakte Reduktion:** Momentum mit β=0 reduziert sich algebraisch exakt auf "
               "SGD:")
    red = _reduces_to_sgd()
    st.metric("Max. Abweichung", f"{red['max_abs_err']:.2e}")

    st.markdown("**Exakte Bias-Korrektur:** Adams $\\hat m_1$ im ersten Schritt ist exakt gleich "
               "$g_1$, unabhängig von $\\beta_1$:")
    bc = _bias_correction()
    st.metric("Max. Abweichung (über 3 getestete β₁)", f"{bc['max_err']:.2e}")

    st.markdown(
        "**Robustheits-Check zu Adams (Nicht-)Vorteil:** selbst unter STARK heterogenem "
        "Rauschen zwischen den beiden Dimensionen (Adams klassische Stärke) bleibt der "
        "Unterschied bei nur zwei Parametern innerhalb der statistischen Schwankung eines "
        "30-Seed-Samples:"
    )
    aniso = _aniso()
    a1, a2 = st.columns(2)
    a1.metric("SGD (bestes η)", f"{aniso['sgd_best']}/{aniso['n_seeds']}")
    a2.metric("Adam (bestes η)", f"{aniso['adam_best']}/{aniso['n_seeds']}")
    st.caption(f"σ_x={aniso['sigma_x']}, σ_y={aniso['sigma_y']} (stark heterogen) — kein "
              "statistisch robuster Vorteil für Adam.")

    grad_err = _gradient_check()
    st.metric("Gradienten-Check", f"{grad_err['max_rel_err']:.1e}")

    st.markdown(
        "**Literatur:** Robbins, H. & Monro, S. (1951). *A stochastic approximation method.* "
        "Annals of Mathematical Statistics, 22(3), 400–407. Polyak, B. T. (1964). *Some methods "
        "of speeding up the convergence of iteration methods.* USSR Comp. Math. and Math. "
        "Physics, 4(5), 1–17. Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic "
        "Optimization.* ICLR 2015. Muon (2024/25): *The Newton-Muon Optimizer*, arXiv:2604.01472 "
        "— hier NICHT gebaut, siehe Ehrliche Vorbehalte im README."
    )
    st.caption(
        "Implementiert in `sgd_functions.py` (verrauschter Rosenbrock-Gradient), "
        "`sgd_optimizers.py` (SGD/Momentum/Adam), `sgd_evaluation.py` (Korrektheits-Kette), "
        "`sgd_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) "
    "– Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung "
    "für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
