#!/usr/bin/env python3
"""Create clearly synthetic PDF mockups for the planned Flex router paper results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mi_lens.plotting.plotly_export import save_plotly_figure


LAYERS = [5, 11, 18, 24, 31]
METHODS = ["Unigram", "Bigram", "Neuron", "PCA", "ITDA", "SAE"]
METHOD_COLORS = {
    "Unigram": "#90A9B5",
    "Bigram": "#64818F",
    "Neuron": "#3E6E7C",
    "PCA": "#287F8C",
    "ITDA": "#4B7586",
    "SAE": "#063F59",
}
LAYER_COLORS = ["#D4E8EB", "#9CCAD0", "#5CA7B1", "#277987", "#063F59"]
PAPER_BG = "#FCFCF8"
INK = "#173943"
GRID = "#DCE7E8"
BLUE_HEATMAP = [
    [0.0, "#F4F8F7"],
    [0.22, "#D9ECEC"],
    [0.5, "#8FC7CE"],
    [0.76, "#3C8E9A"],
    [1.0, "#063F59"],
]
MODEL_LABELS = {
    "flexolmo_7x7b_a2": "FlexOlmo-7x7B-1T-a2",
    "flexolmo_7x7b_a4": "FlexOlmo-7x7B-1T-a4",
    "flexolmo_7x7b_a7": "FlexOlmo-7x7B-1T-a7",
    "flexdanish_8x7b_a2_55b_v2": "FlexDanish-8x7B-1T-a2-55B-v2",
    "flexdanish_8x7b_a4_55b_v2": "FlexDanish-8x7B-1T-a4-55B-v2",
    "flexdanish_8x7b_a7_55b_v2": "FlexDanish-8x7B-1T-a7-55B-v2",
    "flexdanish_8x7b_a8_55b_v2": "FlexDanish-8x7B-1T-a8-55B-v2",
    "flexdanish_8x7b_a4_55b_v2_rt": "FlexDanish-8x7B-1T-a4-55B-v2-RT",
    "flexdanish_8x7b_a8_55b_v2_rt": "FlexDanish-8x7B-1T-a8-55B-v2-RT",
}


def _output_path(value: str) -> Path:
    path = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    tmp_root = (ROOT / "tmp").resolve()
    if path != tmp_root and tmp_root not in path.parents:
        raise ValueError("Mockups must be written under project_root/tmp/.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _style(
    figure: go.Figure,
    title: str | None,
    *,
    height: int = 820,
    has_subplots: bool = False,
) -> None:
    show_legend = any(trace.showlegend is not False and bool(trace.name) for trace in figure.data)
    legend_layout = dict(
        orientation="h", yanchor="bottom", yref="container", y=0.89, xanchor="center", x=0.5
    )
    figure.update_layout(
        template="plotly_white",
        title=(
            dict(
                text=f"<b>{title}</b>", x=0.02, xanchor="left", y=0.99,
                font=dict(size=19, color=INK, family="Arial, sans-serif"),
            )
            if title
            else None
        ),
        # The PDF is placed at 0.96\textwidth: this compact hierarchy remains
        # readable after LaTeX scaling without overpowering the data.
        font=dict(family="Arial, sans-serif", size=18, color=INK),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        width=1280,
        height=height,
        margin=dict(l=78, r=94, t=(48 if show_legend else 40), b=44),
        legend=dict(
            **legend_layout,
            visible=show_legend,
            font=dict(size=16, family="Arial, sans-serif", weight=600), bgcolor="rgba(0,0,0,0)",
        ),
    )
    figure.update_xaxes(
        showline=True, linecolor="#8BA4AA", gridcolor=GRID, zeroline=False,
        title_font=dict(size=17, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=15, family="Arial, sans-serif", weight=600), title_standoff=8,
    )
    figure.update_yaxes(
        showline=True, linecolor="#8BA4AA", gridcolor=GRID, zeroline=False,
        title_font=dict(size=17, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=15, family="Arial, sans-serif", weight=600), title_standoff=8,
    )
    if title:
        # Scale each existing subplot domain into the plot band. This reserves
        # a small title band even when a figure has no legend.
        plot_top = 0.93 if show_legend else 0.94
        for axis in figure.select_yaxes():
            if axis.domain is not None:
                axis.domain = [float(bound) * plot_top for bound in axis.domain]
    for annotation in figure.layout.annotations or ():
        if has_subplots and title:
            plot_top = 0.93 if show_legend else 0.94
            # Keep enlarged subplot titles below the main-title band.
            annotation.y = float(annotation.y) * plot_top - 0.01
        annotation.update(font=dict(size=18, family="Arial, sans-serif", weight=600, color=INK))


def _write(figure: go.Figure, output_dir: Path, stem: str) -> list[str]:
    html = output_dir / f"{stem}.html"
    pdf = output_dir / f"{stem}.pdf"
    save_plotly_figure(figure, html, format="html")
    save_plotly_figure(figure, pdf, format="pdf")
    return [str(html), str(pdf)]


def _predictor_mockup(output_dir: Path) -> list[str]:
    figure = go.Figure()
    bases = {
        "Unigram": 0.42,
        "Bigram": 0.48,
        "Neuron": 0.54,
        "PCA": 0.57,
        "ITDA": 0.61,
        "SAE": 0.66,
    }
    for method_index, method in enumerate(METHODS):
        values = [bases[method] + 0.03 + 0.008 * index + 0.007 * ((method_index + index) % 2) for index in range(5)]
        figure.add_trace(
            go.Scatter(
                x=[f"L{layer}" for layer in LAYERS],
                y=values,
                mode="lines+markers",
                name=method,
                line=dict(color=METHOD_COLORS[method], width=3.2 if method == "SAE" else 2.5 if method == "ITDA" else 2.1),
                marker=dict(size=8),
            )
        )
    figure.update_yaxes(title="Held-out macro-F1", range=[0.3, 0.8])
    figure.update_xaxes(title="Router layer", type="category")
    _style(
        figure,
        "Router prediction baselines: FlexDanish-8x7B-1T-a4-55B-v2",
        height=600,
    )
    figure.update_layout(
        title_font=dict(size=20, family="Arial, sans-serif", weight=600),
        legend=dict(font=dict(size=18, family="Arial, sans-serif", weight=600)),
    )
    figure.update_xaxes(
        title_font=dict(size=19, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=17, family="Arial, sans-serif", weight=600),
    )
    figure.update_yaxes(
        title_font=dict(size=19, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=17, family="Arial, sans-serif", weight=600),
    )
    return _write(figure, output_dir, "01_router_predictors_macro_f1_mock")


def _domain_mockup(output_dir: Path) -> list[str]:
    domains = ["Code", "Creative", "Math", "News", "Academic", "Reddit", "Danish"]
    base_experts = ["Public", "Code", "Creative", "Math", "News", "Academic", "Reddit"]
    danish_experts = [*base_experts, "Danish"]
    base = [
        [0.23, 0.15, 0.16, 0.14, 0.18, 0.17, 0.16],
        [0.47, 0.10, 0.08, 0.06, 0.09, 0.08, 0.05],
        [0.08, 0.50, 0.05, 0.08, 0.07, 0.14, 0.07],
        [0.07, 0.05, 0.54, 0.05, 0.11, 0.04, 0.06],
        [0.05, 0.07, 0.06, 0.52, 0.11, 0.07, 0.06],
        [0.06, 0.04, 0.11, 0.08, 0.49, 0.07, 0.08],
        [0.04, 0.09, 0.05, 0.07, 0.06, 0.43, 0.07],
    ]
    danish = [row[:] for row in base] + [[0.04, 0.05, 0.04, 0.05, 0.07, 0.08, 0.62]]
    danish[0][-1] = 0.10
    for index in range(1, 7):
        danish[index][-1] = 0.04
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("FlexOlmo-7x7B-1T-a4", "FlexDanish-8x7B-1T-a4-55B-v2"),
        horizontal_spacing=0.20,
    )
    for column, matrix, labels in ((1, base, base_experts), (2, danish, danish_experts)):
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=domains,
                y=labels,
                colorscale=BLUE_HEATMAP,
                zmin=0,
                zmax=0.65,
                showscale=column == 2,
                colorbar=dict(
                    title=dict(text="Routing share", font=dict(size=19, family="Arial, sans-serif", weight=600), side="right"),
                    tickvals=[0.0, 0.2, 0.4, 0.6],
                    tickformat=".0%",
                    len=0.72,
                    y=0.5,
                    thickness=18,
                    x=1.03,
                    tickfont=dict(size=18, family="Arial, sans-serif", weight=600),
                )
                if column == 2
                else None,
                hovertemplate="expert=%{y}<br>domain=%{x}<br>share=%{z:.2f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        expected = [
            ("Code", "Code"), ("Creative", "Creative"), ("Math", "Math"),
            ("News", "News"), ("Academic", "Academic"), ("Reddit", "Reddit"),
            ("Danish", "Danish" if "Danish" in labels else "Public"),
        ]
        figure.add_trace(
            go.Scatter(
                x=[domain for domain, _ in expected],
                y=[expert for _, expert in expected],
                mode="markers",
                name="Expected domain expert",
                showlegend=False,
                marker=dict(symbol="square-open", size=17, color="#111111", line=dict(width=1.5, color="#111111")),
                hovertemplate="Expected expert=%{y}<br>domain=%{x}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title="Held-out dataset domain", tickangle=-35, automargin=True, row=1, col=column)
        figure.update_yaxes(title="Expert", autorange="reversed", automargin=True, row=1, col=column)
    _style(
        figure,
        "Observed expert routing by domain",
        height=700,
        has_subplots=True,
    )
    figure.update_layout(title_font=dict(size=21, family="Arial, sans-serif", weight=600))
    figure.update_xaxes(
        title_font=dict(size=19, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=18, family="Arial, sans-serif", weight=600),
    )
    figure.update_yaxes(
        title_font=dict(size=19, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=18, family="Arial, sans-serif", weight=600),
    )
    for axis in figure.select_yaxes():
        axis.domain = [0.0, 0.95]
    for annotation in figure.layout.annotations or ():
        annotation.y = 0.95
        annotation.font = dict(size=20, family="Arial, sans-serif", weight=600, color=INK)
    return _write(figure, output_dir, "02_expert_domain_routing_mock")


def _distribution_mockup(output_dir: Path) -> list[str]:
    """Distributional routing view analogous to the RouterInterp appendices."""

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top-1 routing weight", "Normalised router entropy"),
        horizontal_spacing=0.08,
    )
    models = (
        ("FlexOlmo a4", "#6E98A3", 0.62, 0.56),
        ("FlexDanish a4", "#063F59", 0.66, 0.49),
        ("FlexDanish a4 RT", "#287F8C", 0.59, 0.53),
    )
    for index, (name, color, top1_center, entropy_center) in enumerate(models):
        jitter = [0.09 * (((sample * (index + 3)) % 17) / 16 - 0.5) for sample in range(180)]
        figure.add_trace(
            go.Violin(
                x=[name] * len(jitter), y=[max(0.05, min(0.98, top1_center + value)) for value in jitter],
                name=name, legendgroup=name, showlegend=True, box_visible=True, meanline_visible=True,
                line=dict(color=color, width=2), fillcolor=color, opacity=0.62,
            ),
            row=1, col=1,
        )
        figure.add_trace(
            go.Violin(
                x=[name] * len(jitter), y=[max(0.05, min(0.98, entropy_center - value)) for value in jitter],
                name=name, legendgroup=name, showlegend=False, box_visible=True, meanline_visible=True,
                line=dict(color=color, width=2), fillcolor=color, opacity=0.62,
            ),
            row=1, col=2,
        )
    figure.update_yaxes(title="Probability", range=[0.35, 0.80], row=1, col=1)
    figure.update_yaxes(title="Normalised entropy", range=[0.35, 0.80], row=1, col=2)
    figure.update_xaxes(showticklabels=False, row=1, col=1)
    figure.update_xaxes(showticklabels=False, row=1, col=2)
    _style(figure, "Router-weight distributions across Flex configurations", height=560, has_subplots=True)
    figure.update_layout(
        title=dict(y=0.985, font=dict(size=22, family="Arial, sans-serif", weight=600)),
        legend=dict(y=0.88, font=dict(size=19, family="Arial, sans-serif", weight=600)),
    )
    figure.update_xaxes(
        title_font=dict(size=20, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=18, family="Arial, sans-serif", weight=600),
    )
    figure.update_yaxes(
        title_font=dict(size=20, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=18, family="Arial, sans-serif", weight=600),
    )
    for annotation in figure.layout.annotations or ():
        annotation.y = 0.95
        annotation.font = dict(size=21, family="Arial, sans-serif", weight=600, color=INK)
    for axis in figure.select_yaxes():
        if axis.domain is not None:
            axis.domain = [float(axis.domain[0]), 0.95]
    return _write(figure, output_dir, "05_router_weight_distributions_mock")


def _configuration_mockup(output_dir: Path) -> list[str]:
    labels = [
        "FlexOlmo-7x7B-a2",
        "FlexOlmo-7x7B-a4",
        "FlexOlmo-7x7B-a7",
        "FlexDanish-8x7B-a2-55B-v2",
        "FlexDanish-8x7B-a4-55B-v2",
        "FlexDanish-8x7B-a4-55B-v2-RT",
        "FlexDanish-8x7B-a7-55B-v2",
        "FlexDanish-8x7B-a8-55B-v2",
        "FlexDanish-8x7B-a8-55B-v2-RT",
    ]
    entropy = [0.42, 0.60, 0.92, 0.44, 0.63, 0.58, 0.90, 0.98, 0.94]
    top1 = [0.73, 0.54, 0.29, 0.70, 0.50, 0.55, 0.30, 0.25, 0.27]
    danish_share = [0.0, 0.0, 0.0, 0.17, 0.21, 0.27, 0.16, 0.13, 0.19]
    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Router entropy", "Top-1 routing weight", "Danish-expert selection share"),
        vertical_spacing=0.055,
        shared_xaxes=True,
    )
    colors = ["#6E98A3" if label.startswith("FlexOlmo") else "#063F59" for label in labels]
    for row, values, axis_title in (
        (1, entropy, "Router entropy"),
        (2, top1, "Top-1 weight"),
        (3, danish_share, "Danish selection"),
    ):
        figure.add_trace(
            go.Bar(x=labels, y=values, marker_color=colors, showlegend=False), row=row, col=1
        )
        figure.update_yaxes(title=axis_title, range=[0, 1], row=row, col=1)
    figure.update_xaxes(tickangle=-28, automargin=True, tickfont=dict(size=22), row=3, col=1)
    _style(
        figure,
        "FlexOlmo-7x7B and FlexDanish-8x7B-55B-v2 router configuration summary",
        height=920,
        has_subplots=True,
    )
    # Keep the first panel close to the main title without moving the lower rows.
    figure.update_layout(title=dict(y=0.99, font=dict(size=22, family="Arial, sans-serif", weight=600)))
    figure.update_xaxes(
        title_font=dict(size=20, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=17, family="Arial, sans-serif", weight=600),
    )
    figure.update_yaxes(
        title_font=dict(size=20, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=19, family="Arial, sans-serif", weight=600),
    )
    figure.layout.yaxis.domain = [float(figure.layout.yaxis.domain[0]), 0.935]
    for annotation in figure.layout.annotations or ():
        annotation.font = dict(size=21, family="Arial, sans-serif", weight=600, color=INK)
    figure.layout.annotations[0].y = 0.96
    return _write(figure, output_dir, "03_router_configuration_summary_mock")


def _sae_mockup(output_dir: Path) -> list[str]:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Feature-budget sensitivity", "SAE feature health", "Feature/router agreement"),
        horizontal_spacing=0.08,
    )
    budgets = ["1", "2", "4", "8", "16", "32"]
    for name, offset, color in (("FlexOlmo-7x7B-1T-a4", 0.0, "#4F8A96"), ("FlexDanish-8x7B-1T-a4-55B-v2", 0.035, "#063F59")):
        figure.add_trace(
            go.Scatter(x=budgets, y=[0.47 + offset, 0.52 + offset, 0.58 + offset, 0.62 + offset, 0.65 + offset, 0.66 + offset], mode="lines+markers", name=name, line=dict(color=color, width=3)),
            row=1, col=1,
        )
    figure.add_trace(go.Bar(x=[f"L{layer}" for layer in LAYERS], y=[0.28, 0.21, 0.16, 0.19, 0.25], marker_color="#287F8C", name="Dead features"), row=1, col=2)
    figure.add_trace(go.Scatter(x=[f"L{layer}" for layer in LAYERS], y=[0.31, 0.44, 0.51, 0.46, 0.39], mode="lines+markers", marker_color="#063F59", line=dict(width=3), name="Spearman $\\rho$"), row=1, col=3)
    figure.update_xaxes(title="Retained SAE latents", type="category", row=1, col=1)
    figure.update_yaxes(title="Held-out macro-F1", range=[0.35, 0.75], row=1, col=1)
    figure.update_xaxes(title="Router layer", row=1, col=2)
    figure.update_yaxes(title="Dead feature fraction", range=[0, 1], row=1, col=2)
    figure.update_xaxes(title="Router layer", row=1, col=3)
    figure.update_yaxes(title="Spearman correlation", range=[-1, 1], row=1, col=3)
    _style(
        figure,
        "Sparse-feature diagnostics",
        height=650,
        has_subplots=True,
    )
    figure.update_layout(
        title=dict(y=0.985, font=dict(size=22, family="Arial, sans-serif", weight=600)),
        legend=dict(y=0.89, font=dict(size=19, family="Arial, sans-serif", weight=600)),
    )
    figure.update_xaxes(
        title_font=dict(size=20, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=17, family="Arial, sans-serif", weight=600),
    )
    figure.update_yaxes(
        title_font=dict(size=20, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=17, family="Arial, sans-serif", weight=600),
    )
    for axis in figure.select_yaxes():
        if axis.domain is not None:
            axis.domain = [float(axis.domain[0]), 0.96]
    for annotation in figure.layout.annotations or ():
        annotation.y = 0.95
        annotation.font = dict(size=21, family="Arial, sans-serif", weight=600, color=INK)
    return _write(figure, output_dir, "04_sae_router_diagnostics_mock")


def _model_summary_mockups(output_dir: Path) -> list[str]:
    """Create one appendix-style router summary layout for every configuration."""

    paths: list[str] = []
    domains = ["Code", "Creative", "Math", "News", "Academic", "Reddit", "Danish"]
    for index, (model_key, model_label) in enumerate(MODEL_LABELS.items()):
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Routing-predictor macro-F1",
                "Router concentration",
                "Observed expert routing",
                "Sparse-feature health",
            ),
            horizontal_spacing=0.16,
            vertical_spacing=0.15,
        )
        layers = [f"L{layer}" for layer in LAYERS]
        model_offset = 0.004 * (index % 4)
        for name, base, color in (
            ("PCA", 0.57, "#287F8C"),
            ("ITDA", 0.61, "#4B7586"),
            ("SAE", 0.66, "#063F59"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=layers,
                    y=[base + model_offset + 0.006 * layer_index for layer_index in range(len(layers))],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    line=dict(color=color, width=3 if name == "SAE" else 2.4),
                    marker=dict(size=8),
                ),
                row=1,
                col=1,
            )
        figure.add_trace(
            go.Bar(
                x=layers,
                y=[0.48 + 0.06 * ((index + layer_index) % 4) for layer_index in range(len(layers))],
                name="Top-1 weight",
                marker_color="#5C98A5",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        experts = ["Public", "Code", "Creative", "Math", "News", "Academic", "Reddit"]
        if model_key.startswith("flexdanish"):
            experts.append("Danish")
        routing = []
        for expert_index in range(len(experts)):
            row = [0.05 + 0.01 * ((expert_index + domain_index + index) % 4) for domain_index in range(len(domains))]
            row[(expert_index + index) % len(domains)] = 0.44 + 0.03 * (index % 3)
            routing.append(row)
        figure.add_trace(
            go.Heatmap(
                z=routing,
                x=domains,
                y=experts,
                colorscale=BLUE_HEATMAP,
                zmin=0,
                zmax=0.55,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Routing share", side="right", font=dict(size=20, family="Arial, sans-serif", weight=600)),
                    tickvals=[0.0, 0.2, 0.4],
                    tickformat=".0%",
                    tickfont=dict(size=18, family="Arial, sans-serif", weight=600),
                    thickness=16,
                    len=0.26,
                    y=0.205,
                    x=0.44,
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=layers,
                y=[0.24, 0.20, 0.16, 0.19, 0.23],
                name="Dead features",
                marker_color="#287F8C",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        figure.update_xaxes(title=None, row=1, col=1)
        figure.update_yaxes(title="Macro-F1", range=[0.45, 0.75], row=1, col=1)
        figure.update_xaxes(title=None, row=1, col=2)
        figure.update_yaxes(title="Top-1 weight", range=[0, 1], row=1, col=2)
        figure.update_xaxes(title="Dataset domain", tickangle=-30, row=2, col=1)
        figure.update_yaxes(title="Expert", autorange="reversed", row=2, col=1)
        figure.update_xaxes(title="Router layer", row=2, col=2)
        figure.update_yaxes(title="Dead-feature share", range=[0, 1], row=2, col=2)
        figure.update_yaxes(side="right", row=2, col=2)
        _style(figure, f"{model_label}: router summary", height=820, has_subplots=True)
        figure.update_layout(
            title=dict(y=0.985, font=dict(size=22, family="Arial, sans-serif", weight=600)),
            legend=dict(y=0.92, font=dict(size=19, family="Arial, sans-serif", weight=600)),
        )
        figure.update_xaxes(
            title_font=dict(size=20, family="Arial, sans-serif", weight=600),
            tickfont=dict(size=17, family="Arial, sans-serif", weight=600),
        )
        figure.update_yaxes(
            title_font=dict(size=20, family="Arial, sans-serif", weight=600),
            tickfont=dict(size=17, family="Arial, sans-serif", weight=600),
        )
        figure.layout.yaxis.domain = [0.54, 0.96]
        figure.layout.yaxis2.domain = [0.54, 0.96]
        figure.layout.yaxis3.domain = [0.06, 0.35]
        figure.layout.yaxis4.domain = [0.06, 0.35]
        for annotation_index, annotation in enumerate(figure.layout.annotations or ()):
            annotation.y = 0.97 if annotation_index < 2 else 0.40
            annotation.font = dict(size=21, family="Arial, sans-serif", weight=600, color=INK)
        paths.extend(_write(figure, output_dir, f"06_router_summary_{model_key}_mock"))
    return paths


def _write_tables(output_dir: Path) -> list[str]:
    predictors = output_dir / "table_router_predictors_mock.tex"
    predictors.write_text(
        """% ILLUSTRATIVE MOCKUP -- synthetic values, not measured results.
\\begin{tabular}{llrrrrrr}
\\toprule
Model & Layers & Unigram & Bigram & Neuron & PCA & ITDA & SAE \\\\
\\midrule
FlexOlmo-7x7B-1T-a2 & L5--L31 & 0.42 & 0.48 & 0.54 & 0.57 & 0.61 & 0.66 \\\\
FlexOlmo-7x7B-1T-a4 & L5--L31 & 0.39 & 0.45 & 0.51 & 0.55 & 0.59 & 0.62 \\\\
FlexDanish-8x7B-1T-a2-55B-v2 & L5--L31 & 0.44 & 0.50 & 0.56 & 0.59 & 0.63 & 0.68 \\\\
FlexDanish-8x7B-1T-a4-55B-v2 & L5--L31 & 0.43 & 0.49 & 0.55 & 0.58 & 0.62 & 0.67 \\\\
FlexDanish-8x7B-1T-a4-55B-v2-RT & L5--L31 & 0.45 & 0.51 & 0.57 & 0.60 & 0.65 & 0.70 \\\\
\\bottomrule
\\end{tabular}
""",
        encoding="utf-8",
    )
    dynamics = output_dir / "table_router_dynamics_mock.tex"
    dynamics.write_text(
        """% ILLUSTRATIVE MOCKUP -- synthetic values, not measured results.
\\begin{tabular}{lrrr}
\\toprule
Configuration & Router entropy & Top-1 weight & Danish-expert share \\\\
\\midrule
FlexOlmo-7x7B-1T-a2 & 0.42 & 0.73 & -- \\\\
FlexOlmo-7x7B-1T-a4 & 0.60 & 0.54 & -- \\\\
FlexOlmo-7x7B-1T-a7 & 0.92 & 0.29 & -- \\\\
FlexDanish-8x7B-1T-a2-55B-v2 & 0.44 & 0.70 & 0.17 \\\\
FlexDanish-8x7B-1T-a4-55B-v2 & 0.63 & 0.50 & 0.21 \\\\
FlexDanish-8x7B-1T-a4-55B-v2-RT & 0.58 & 0.55 & 0.27 \\\\
FlexDanish-8x7B-1T-a7-55B-v2 & 0.90 & 0.30 & 0.16 \\\\
FlexDanish-8x7B-1T-a8-55B-v2 & 0.98 & 0.25 & 0.13 \\\\
FlexDanish-8x7B-1T-a8-55B-v2-RT & 0.94 & 0.27 & 0.19 \\\\
\\bottomrule
\\end{tabular}
""",
        encoding="utf-8",
    )
    return [str(predictors), str(dynamics)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/routerinterp/mockups_v3", help="Directory under project tmp/.")
    args = parser.parse_args()
    output_dir = _output_path(args.output)
    artifacts = [
        *_predictor_mockup(output_dir),
        *_domain_mockup(output_dir),
        *_configuration_mockup(output_dir),
        *_sae_mockup(output_dir),
        *_distribution_mockup(output_dir),
        *_model_summary_mockups(output_dir),
        *_write_tables(output_dir),
    ]
    manifest = {
        "warning": "All figures and tables in this directory use synthetic mock values only.",
        "models": MODEL_LABELS,
        "selected_layers": LAYERS,
        "artifacts": artifacts,
    }
    (output_dir / "mockup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
