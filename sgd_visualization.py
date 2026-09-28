"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen fest uebergeben (siehe
feedback_plotly_fixedrange_convention/feedback_plotly_scaleanchor_explicit_range). Kein
literales '|' in Markdown-Tabellenzellen (feedback_markdown_table_literal_pipe_breaks_columns)."""
import numpy as np
import plotly.graph_objects as go

import sgd_functions as fn

COLOR_SGD = "#1f77b4"
COLOR_MOMENTUM = "#d62728"
COLOR_ADAM = "#2ca02c"
COLOR_OPT = "#9467bd"


def build_trajectory_figure(trajectories: dict, r_range=(-1.6, 1.6), h_range=(-0.6, 1.6),
                            title=""):
    xs = np.linspace(r_range[0], r_range[1], 150)
    ys = np.linspace(h_range[0], h_range[1], 150)
    Z = np.zeros((len(ys), len(xs)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            Z[i, j] = np.log10(1 + fn.rosenbrock(np.array([xv, yv])))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale="Blues", contours=dict(coloring="fill"),
        opacity=0.5,
    ))
    colors = {"SGD": COLOR_SGD, "Momentum": COLOR_MOMENTUM, "Adam": COLOR_ADAM}
    for name, res in trajectories.items():
        traj_x = [p[0] for p in res.trajectory]
        traj_y = [p[1] for p in res.trajectory]
        fig.add_trace(go.Scatter(x=traj_x, y=traj_y, mode="lines", name=name,
                                 line=dict(color=colors.get(name, "#333"), width=2)))
    fig.add_trace(go.Scatter(x=[fn.X_STAR[0]], y=[fn.X_STAR[1]], mode="markers",
                             name="Minimum (1,1)",
                             marker=dict(color=COLOR_OPT, size=14, symbol="star")))
    fig.update_layout(
        title=title, xaxis=dict(range=list(r_range), fixedrange=True, title="x"),
        yaxis=dict(range=list(h_range), fixedrange=True, title="y"),
        showlegend=True, height=440, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_noise_sweep_figure(rows, title="Erfolgsquote vs. Rauschstärke σ"):
    sigmas = [r["sigma"] for r in rows]
    n_seeds = rows[0]["n_seeds"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sigmas, y=[r["sgd"] for r in rows], mode="lines+markers",
                             name="SGD", line=dict(color=COLOR_SGD, width=2)))
    fig.add_trace(go.Scatter(x=sigmas, y=[r["momentum"] for r in rows], mode="lines+markers",
                             name="Momentum", line=dict(color=COLOR_MOMENTUM, width=2)))
    fig.add_trace(go.Scatter(x=sigmas, y=[r["adam"] for r in rows], mode="lines+markers",
                             name="Adam", line=dict(color=COLOR_ADAM, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Rauschstärke σ", fixedrange=True),
        yaxis=dict(title=f"Erfolge (von {n_seeds} Seeds)", fixedrange=True, range=[0, n_seeds]),
        showlegend=True, height=380, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
