# Stochastische Gradientenverfahren – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-stochastische-gradientenverfahren-demo.streamlit.app/)**

Stück 8 (LETZTES) der **Nichtlineare-Optimierung-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Stück 2–7 behoben fehlende Krümmungsinformation (Newton-Ast). Dieses Stück ist der ZWEITE Ast
direkt von der Wurzel (Gradientenabstieg) und behebt eine ANDERE Ignoranz: **teure/verrauschte
Gradienten** bei großem $n$ — genau die Situation beim Training eines neuronalen Netzes
(Mini-Batch statt vollem Datensatz). Das ist zugleich die Brücke zu Deep Learning und der
Abschluss der gesamten Linie.

**Einordnung in die Reihe:**

```
Gradientenabstieg (WURZEL)                       [gebaut]
 └─ Newton-Verfahren                             [gebaut]
      └─ Quasi-Newton (BFGS/L-BFGS)               [gebaut]
           └─ Lagrange-Multiplikatoren/KKT        [gebaut]
                ├─ Straf-/Barriere-Verfahren      [gebaut]
                └─ SQP                            [gebaut]
                     └─ Innere-Punkte-Verfahren   [gebaut]
 └─ Stochastische Gradientenverfahren             [DIESES STÜCK — LINIE VOLLSTÄNDIG]
```

**Vehikel C (neu, bewusst KEIN neuronales Netz):** eine verrauschte Variante von Vehikel A —
dieselbe Rosenbrock-Funktion aus Stück 1–3 (eigene Kopie), aber bei jedem Schritt bekommt der
Gradient gaußsches Rauschen: $g_{\text{noisy}}(x)=\nabla f(x)+\sigma\varepsilon$. Damit steht der
Optimierer selbst im Zentrum, nicht ein konkretes Netz.

**Ergebnis in Kürze:** die Hypothese "Momentum ist bei gleicher Lernrate schlechter als SGD"
(repliziert von `mlp-backprop-demo`, dort 0/10 gegen 7/10 auf XOR) hält NICHT unverändert stand —
sie wird VERFEINERT: rauschfrei ist Momentum hier sogar besser (klassische Beschleunigung). Erst
mit wachsendem Gradientenrauschen bricht seine Erfolgsquote massiv stärker ein als bei SGD (bei
σ=20: SGD 14/30 gegen Momentum 1/30). **Adam zeigt hier — anders als bei `mlp-backprop-demo`
(9/10 gegen 7/10) — keinen klaren Vorteil gegenüber SGD**, auch nicht unter stark heterogenem
Rauschen: bei nur zwei Parametern gleicht eine gut getunte einzelne Schrittweite Adams
Pro-Parameter-Anpassung fast vollständig aus. Muon wurde deshalb NICHT gebaut (Vormessung zeigte
keinen echten Unterschied) und bleibt Literatur-Ausblick.

## Warum dieses Problem

Die gesamte Linie behandelte bisher EXAKTE Gradienten. Reale Optimierung bei großen Datenmengen
(und jedes Training eines neuronalen Netzes) hat das nie — Gradienten kommen aus Stichproben
(Mini-Batches) und sind damit verrauscht. Dieses Stück zeigt, wie sich die aus der
Deep-Learning-Praxis bekannten Optimierer (SGD, Momentum, Adam) unter genau dieser Bedingung auf
einer generischen, gut verstandenen Testfunktion verhalten — und schließt damit den Kreis zur
Neuronale-Netze-Linie.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Momentum ist bei gleicher Lernrate schlechter als SGD (Replikation `mlp-backprop-demo`) | ⚠️ **Verfeinert:** rauschfrei ist Momentum BESSER (klassische Beschleunigung); erst unter Rauschen kehrt sich das um |
| Adams Vorteil gegenüber SGD wächst mit dem Rauschen | ⚠️ **Widerlegt** bei dieser Dimension (2 Parameter) — Adam bleibt praktisch gleichauf mit SGD über den ganzen Rauschbereich, auch unter heterogenem Rauschen |
| Rauschfreie Baseline: alle drei erreichen dasselbe Minimum wie in Stück 1 | ✅ SGD/Adam nah dran (0,165 / 0,0012), Momentum praktisch exakt ($2{,}9\cdot10^{-9}$) |
| Momentum mit β=0 reduziert sich exakt auf SGD | ✅ Abweichung $0{,}0$ |
| Adams bias-korrigiertes erstes Moment ist im ersten Schritt exakt der Gradient | ✅ Abweichung $\le2{,}8\cdot10^{-14}$ |
| Gradienten-Check unter $10^{-6}$ | ✅ $4{,}3\cdot10^{-11}$ |

## Befunde (gemessen, keine Behauptungen)

**Erfolgsquote vs. Rauschstärke** (30 Seeds, 3000 Iterationen, $\eta_{\text{SGD/Mom}}=0{,}0015$,
$\eta_{\text{Adam}}=0{,}02$, $\beta_{\text{Mom}}=0{,}9$, Toleranz 0,5 zum Minimum):

| σ | SGD | Momentum | Adam |
|---|---|---|---|
| 0 | 30/30 | 30/30 | 30/30 |
| 2 | 30/30 | 30/30 | 30/30 |
| 5 | 29/30 | 20/30 | 29/30 |
| 10 | 20/30 | 10/30 | 20/30 |
| 20 | 14/30 | 1/30 | 14/30 |
| 50 | 4/30 | 0/30 | 2/30 |

SGD und Adam laufen über den GESAMTEN Rauschbereich nahezu identisch — Momentum bricht ab σ=5
deutlich stärker ein und kollabiert bei σ≥20 praktisch vollständig.

**Robustheits-Check zu Adams Nicht-Vorteil** (stark heterogenes Rauschen zwischen den Dimensionen,
$\sigma_x=2$, $\sigma_y=40$, je bestmöglich getunte Lernrate, 30 Seeds): SGD 16/30, Adam 17/30 —
innerhalb der statistischen Schwankung eines 30-Seed-Samples, kein robuster Vorteil, selbst unter
der Bedingung, die Adams Pro-Parameter-Anpassung eigentlich begünstigen sollte.

**Warum kein Adam-Vorteil hier, aber bei `mlp-backprop-demo` schon (9/10 gegen 7/10)?** Adams
Stärke ist die Anpassung an HETEROGENE Gradientenskalen über VIELE Parameter. Ein neuronales Netz
hat dutzende bis tausende Gewichte mit sehr unterschiedlicher Krümmung/Skala — dort lohnt sich
Pro-Parameter-Anpassung deutlich. Bei nur 2 Parametern kann eine von Hand gut getunte einzelne
Schrittweite fast denselben Effekt erzielen. Eine echte, dimensionsabhängige Grenze des Vehikels,
nicht des Verfahrens.

**Exakte Reduktion:** Momentum mit $\beta=0$ reproduziert SGD bit-identisch (Abweichung $0{,}0$).

**Exakte Bias-Korrektur:** $\hat m_1=m_1/(1-\beta_1)=g_1$ unabhängig von $\beta_1$ — numerisch auf
$2{,}8\cdot10^{-14}$ bestätigt.

## Modell und Verfahren

- `sgd_functions.py` – Rosenbrock-Funktion + Gradient (Kopie aus Stück 1–3), plus
  `noisy_gradient()`.
- `sgd_optimizers.py` – `sgd_run`, `momentum_run`, `adam_run` (je mit fester Seed-Steuerung und
  Trajektorien-Aufzeichnung), `success_rate()`.
- `sgd_evaluation.py` – Momentum-Rausch-Anfälligkeit, Adams (Nicht-)Vorteil (isotrop UND
  anisotrop), rauschfreie Baseline, exakte Reduktions-/Bias-Korrektur-Checks, Gradienten-Check.
- `sgd_visualization.py` – Plotly: Trajektorien auf Rosenbrock-Höhenlinien, Erfolgsquote-vs-σ.

## Was die App zeigt

Rauschstärke σ in der Sidebar; eine illustrative Einzel-Trajektorie aller drei Verfahren auf den
Rosenbrock-Höhenlinien; die Erfolgsquote-vs-σ-Messreihe als zentrale Messung (30 Seeds, alle drei
Verfahren); ein "📐"-Abschnitt mit der vollständigen Korrektheits-Kette inkl. des
Robustheits-Checks zu Adams Nicht-Vorteil.

## Was nicht funktioniert hat / Grenzen

**Zwei echte Plan-Korrekturen:** (1) "Momentum schlechter als SGD" gilt NICHT pauschal, sondern
nur unter Gradientenrauschen — rauschfrei ist Momentum klassisch schneller. (2) "Adams Vorteil
wächst mit dem Rauschen" wurde bei dieser (bewusst kleinen) Dimensionalität WIDERLEGT, selbst mit
gezielt heterogenem Rauschen — der Vorteil braucht mehr Parameter, um sich zu zeigen (siehe
`mlp-backprop-demo`).

**Muon nicht gebaut:** die Vormessung zeigte keinen echten, stabilen Unterschied zu Adam auf
diesem 2-Parameter-Vehikel (konsistent mit Befund 2) — bleibt reiner Literatur-Ausblick, wie
Mamba am Ende der Neuronale-Netze-Linie.

**Grenzen:** nur zwei Parameter (bewusst — hält die Vehikel-Größe im Rahmen der übrigen Linie).
Isotropes Grundrauschen (Robustheits-Check mit anisotropem Rauschen ergänzt, aber nicht als
Regler in der App).

## Tests

34 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ~40 Sekunden — die Rausch-Sweeps über 30
Seeds × 3000 Iterationen dominieren):
- `test_functions.py` – Rosenbrock-Funktion/Gradient, Rauschen reproduzierbar mit festem Seed.
- `test_optimizers.py` – Konvergenz rauschfrei, exakte Momentum-Reduktion, Trajektorien-Länge.
- `test_evaluation.py`, `test_claims.py` – jede Zahl oben nachgerechnet, plattformrobust (Richtung
  statt exakter Zählwert, siehe `feedback_ci_platform_robust_tests`).
- `test_presets.py`, `test_app.py` – Presets, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `sgd_constants.py` | Regler-Grenzen, feste Lernraten, Presets |
| `sgd_functions.py` | Rosenbrock-Funktion, verrauschter Gradient |
| `sgd_optimizers.py` | SGD, Momentum, Adam |
| `sgd_evaluation.py` | Korrektheits-Kette |
| `sgd_visualization.py` | Plotly-Plots |
| `sgd_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Muon (2024/25) — Vormessung zeigte keinen stabilen Vorteil auf diesem kleinen Vehikel, siehe oben.
Kein Lernraten-Scheduler (bewusst — fixe Lernraten halten den Vergleich fair und einfach
nachvollziehbar). Kein echtes neuronales Netz (bewusst, siehe Vehikel-Begründung oben — Crosslink
zu `mlp-backprop-demo` statt Dopplung).

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Robbins, H. & Monro, S. (1951). *A stochastic approximation method.* Annals of Mathematical
  Statistics, 22(3), 400–407.
- Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods.* USSR
  Computational Mathematics and Mathematical Physics, 4(5), 1–17.
- Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015.
- Muon (2024/25, nicht gebaut, siehe oben): *The Newton-Muon Optimizer*, arXiv:2604.01472.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html).
