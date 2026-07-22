"""Publication-oriented reports for direct RouterInterp routing evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plotly_export import save_plotly_figure


_METHOD_LABELS = {
    "unigram_baseline": "Unigram baseline",
    "bigram_baseline": "Bigram baseline",
    "neuron_basis_probe": "Neuron basis probe",
    "pca_basis_probe": "PCA basis probe",
    "sae_predictor": "SAE predictor",
    "itda_predictor": "ITDA predictor",
}
_METHOD_COLORS = {
    "Unigram baseline": "#90A9B5",
    "Bigram baseline": "#64818F",
    "Neuron basis probe": "#3E6E7C",
    "PCA basis probe": "#287F8C",
    "SAE predictor": "#063F59",
    "ITDA predictor": "#4B7586",
}
_LAYER_COLORS = ["#D4E8EB", "#9CCAD0", "#5CA7B1", "#277987", "#063F59"]
_PAPER_BG = "#FCFCF8"
_INK = "#173943"
_GRID = "#DCE7E8"
_BLUE_HEATMAP = [
    [0.0, "#F4F8F7"],
    [0.22, "#D9ECEC"],
    [0.5, "#8FC7CE"],
    [0.76, "#3C8E9A"],
    [1.0, "#063F59"],
]
_MODEL_DISPLAY_NAMES = {
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
_UNIT_INTERVAL_METRICS = {
    "set_precision_at_k",
    "set_recall_at_k",
    "set_jaccard_at_k",
    "macro_f1",
    "jensen_shannon_divergence",
    "total_variation_distance",
    "selection_brier_score",
    "selection_ece",
}


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_") or "model"


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _validate_metric(name: str, value: object, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}: {name} must be finite, found {value!r}.")
    if name in _UNIT_INTERVAL_METRICS and not 0.0 <= number <= 1.0:
        raise ValueError(f"{context}: {name} must be in [0, 1], found {number}.")
    if name == "kl_actual_to_predicted" and number < 0.0:
        raise ValueError(f"{context}: KL divergence must be non-negative, found {number}.")
    return number


def _model_label(summary_path: Path, payload: dict[str, Any]) -> str:
    configured = str(payload.get("model_label", "")).strip()
    raw_label = configured or summary_path.parent.parent.name
    return _MODEL_DISPLAY_NAMES.get(raw_label, raw_label)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Keep a stable union when methods expose different optional diagnostics."""

    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _write_figure(
    figure: go.Figure,
    path: Path,
    title: str,
    *,
    height: int = 610,
    has_subplots: bool = False,
    legend_font_size: int = 14,
) -> None:
    """Export one compact, paper-readable Plotly figure via Kaleido."""

    show_legend = any(trace.showlegend is not False and bool(trace.name) for trace in figure.data)
    figure.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>{title}</b>", x=0.02, xanchor="left", y=0.985,
            font=dict(size=20, color=_INK, family="Arial, sans-serif"),
        ),
        font=dict(family="Arial, sans-serif", size=15, color=_INK),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PAPER_BG,
        width=1280,
        height=height,
        margin=dict(l=92, r=108, t=(72 if show_legend else 58), b=68),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            yref="container",
            y=0.925,
            xanchor="center",
            x=0.5,
            visible=show_legend,
            font=dict(size=legend_font_size, family="Arial, sans-serif", weight=600),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    figure.update_xaxes(
        showline=True,
        linecolor="#8BA4AA",
        gridcolor=_GRID,
        zeroline=False,
        title_font=dict(size=15, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=12, family="Arial, sans-serif", weight=600),
        title_standoff=10,
    )
    figure.update_yaxes(
        showline=True,
        linecolor="#8BA4AA",
        gridcolor=_GRID,
        zeroline=False,
        title_font=dict(size=15, family="Arial, sans-serif", weight=600),
        tickfont=dict(size=12, family="Arial, sans-serif", weight=600),
        title_standoff=10,
    )
    if title:
        # Reserve a compact title band for all figures. Figures with a legend
        # use a slightly lower plot top so the legend has its own row.
        plot_top = 0.89 if show_legend else 0.92
        for axis in figure.select_yaxes():
            if axis.domain is not None:
                axis.domain = [float(bound) * plot_top for bound in axis.domain]
    for annotation in figure.layout.annotations or ():
        if has_subplots and title:
            plot_top = 0.89 if show_legend else 0.92
            annotation.y = float(annotation.y) * plot_top + 0.015
        annotation.update(font=dict(size=17, family="Arial, sans-serif", weight=600, color=_INK))
    save_plotly_figure(figure, path, format="html")
    save_plotly_figure(figure, path.with_suffix(".pdf"), format="pdf")


def _collect_summary(
    summary_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("format") != "mi_lens.routerinterp.analysis.v1":
        raise ValueError(f"{summary_path} is not a RouterInterp analysis summary.")
    model = _model_label(summary_path, payload)
    routing_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for layer_text, layer_result in sorted(payload.get("layers", {}).items(), key=lambda item: int(item[0])):
        layer = int(layer_text)
        context = f"{model}, layer {layer}"
        routing = layer_result["routing_prediction"]
        for method_key, method_label in _METHOD_LABELS.items():
            metrics = routing.get(method_key)
            if metrics is None:
                continue
            row = {"model": model, "layer": layer, "method": method_label}
            for name, value in metrics.items():
                row[name] = _validate_metric(name, value, context=context)
            routing_rows.append(row)
        for active_features, metrics in routing["sae_predictor_by_active_features"].items():
            row = {"model": model, "layer": layer, "active_features": int(active_features)}
            for name, value in metrics.items():
                row[name] = _validate_metric(name, value, context=context)
            budget_rows.append(row)
        diagnostics = layer_result["feature_activation_diagnostics"]
        health_rows.append(
            {
                "model": model,
                "layer": layer,
                **{name: _validate_metric(name, value, context=context) if "fraction" in name or "rate" in name else value
                   for name, value in diagnostics.items()},
            }
        )
        activation_views = (
            ("Observed router", layer_result.get("domain_expert_activation")),
            ("SAE predictor", layer_result.get("sae_predicted_domain_expert_activation")),
        )
        for view, activation in activation_views:
            if not activation:
                continue
            for group in activation["groups"]:
                rates = group["expert_activation_rate"]
                if len(rates) != len(layer_result["expert_labels"]):
                    raise ValueError(f"{context}: {view} expert rates do not match the configured labels.")
                for expert, rate in enumerate(rates):
                    domain_rows.append(
                        {
                            "model": model,
                            "layer": layer,
                            "view": view,
                            "expert": expert,
                            "expert_label": layer_result["expert_labels"][expert],
                            "domain": str(group["group"]),
                            "routing_share": _validate_metric("routing_share", rate, context=context),
                        }
                    )
        if not any(row["model"] == model and row["layer"] == layer for row in domain_rows):
            # Backward-compatible reader for reports generated before the direct
            # observed-versus-predicted domain comparison was introduced.
            domain_routing = layer_result["domain_routing"]
            for expert in domain_routing["per_expert"]:
                for domain, share in expert["domain_routing_share"].items():
                    domain_rows.append(
                        {
                            "model": model,
                            "layer": layer,
                            "view": "Observed router (legacy P(domain | expert))",
                            "expert": int(expert["expert"]),
                            "expert_label": layer_result["expert_labels"][int(expert["expert"])],
                            "domain": domain,
                            "routing_share": _validate_metric("routing_share", share, context=context),
                        }
                    )
        for metric, result in layer_result.get("router_probability_distribution", {}).items():
            if metric == "token_count":
                continue
            histogram = result["histogram"]
            edges = [float(value) for value in histogram["edges"]]
            counts = [int(value) for value in histogram["counts"]]
            if len(edges) != len(counts) + 1:
                raise ValueError(f"{context}: {metric} histogram edges and counts are inconsistent.")
            if (
                not all(math.isfinite(value) for value in edges)
                or edges[0] != 0.0
                or edges[-1] != 1.0
                or any(right <= left for left, right in zip(edges, edges[1:]))
            ):
                raise ValueError(f"{context}: {metric} histogram edges must be finite and span [0, 1].")
            if any(count < 0 for count in counts):
                raise ValueError(f"{context}: {metric} histogram contains a negative count.")
            total = sum(counts)
            if total <= 0:
                raise ValueError(f"{context}: {metric} histogram is empty.")
            for index, count in enumerate(counts):
                distribution_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "metric": metric,
                        "bin_left": edges[index],
                        "bin_right": edges[index + 1],
                        "count": count,
                        "density": count / total,
                    }
                )
    return routing_rows, budget_rows, health_rows, domain_rows, distribution_rows


def _write_predictor_table(rows: list[dict[str, Any]], path: Path) -> None:
    header = "Model & Layer & Method & Macro-F1 & Precision@k & Recall@k & Jaccard@k & JSD & TV \\\\"
    body = []
    for row in rows:
        body.append(
            " & ".join(
                (
                    _latex_escape(row["model"]),
                    f"L{row['layer']}",
                    _latex_escape(row["method"]),
                    f"{row['macro_f1']:.3f}",
                    f"{row['set_precision_at_k']:.3f}",
                    f"{row['set_recall_at_k']:.3f}",
                    f"{row['set_jaccard_at_k']:.3f}",
                    f"{row['jensen_shannon_divergence']:.3f}",
                    f"{row['total_variation_distance']:.3f}",
                )
            )
            + r" \\"
        )
    path.write_text(
        "\n".join(
            (
                r"\begin{tabular}{lllrrrrrr}",
                r"\toprule",
                header,
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _mean_and_sample_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _write_main_predictor_table(rows: list[dict[str, Any]], path: Path) -> None:
    """Compact main-paper table: held-out macro-F1 averaged over chosen layers."""

    grouped: dict[tuple[str, str], list[float]] = {}
    layers: dict[str, set[int]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["method"]), []).append(float(row["macro_f1"]))
        layers.setdefault(row["model"], set()).add(int(row["layer"]))
    models = sorted(layers)
    methods = [
        "Unigram baseline",
        "Bigram baseline",
        "Neuron basis probe",
        "PCA basis probe",
        "SAE predictor",
        "ITDA predictor",
    ]
    header = "Model & Layers & " + " & ".join(methods) + r" \\"
    body = []
    for model in models:
        values = []
        for method in methods:
            if (model, method) not in grouped:
                values.append("--")
                continue
            mean, std = _mean_and_sample_std(grouped[(model, method)])
            values.append(f"{mean:.3f} $\\pm$ {std:.3f}")
        body.append(
            " & ".join(
                (_latex_escape(model), ", ".join(f"L{layer}" for layer in sorted(layers[model])), *values)
            )
            + r" \\"
        )
    path.write_text(
        "\n".join(
            (
                r"\begin{tabular}{llrrrrrr}",
                r"\toprule",
                header,
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _by_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    return grouped


def _predictor_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Keep each configuration separate: five method traces remain legible."""

    paths: list[str] = []
    for model, model_rows in sorted(_by_model(rows).items()):
        figure = go.Figure()
        for method in _METHOD_LABELS.values():
            group = sorted((row for row in model_rows if row["method"] == method), key=lambda row: row["layer"])
            if not group:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[f"L{row['layer']}" for row in group],
                    y=[row["macro_f1"] for row in group],
                    mode="lines+markers",
                    name=method,
                    showlegend=True,
                    line=dict(color=_METHOD_COLORS[method], width=3.2 if method == "SAE predictor" else 2.1),
                    marker=dict(size=8),
                    hovertemplate="%{fullData.name}<br>%{x}<br>macro-F1=%{y:.3f}<extra></extra>",
                )
            )
        figure.update_yaxes(title="Held-out macro-F1", range=[0, 1], tickformat=".1f")
        figure.update_xaxes(title="Transformer layer", type="category")
        destination = output_dir / f"figure_router_predictors_macro_f1_{_safe_label(model)}.html"
        _write_figure(
            figure,
            destination,
            "Router-prediction macro-F1",
        )
        paths.extend((str(destination), str(destination.with_suffix(".pdf"))))
    return paths


def _budget_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    paths: list[str] = []
    for model, model_rows in sorted(_by_model(rows).items()):
        figure = go.Figure()
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for row in model_rows:
            by_layer.setdefault(int(row["layer"]), []).append(row)
        for index, (layer, group) in enumerate(sorted(by_layer.items())):
            group.sort(key=lambda row: row["active_features"])
            figure.add_trace(
                go.Scatter(
                    x=[str(row["active_features"]) for row in group],
                    y=[row["macro_f1"] for row in group],
                    mode="lines+markers",
                    name=f"L{layer}",
                    line=dict(color=_LAYER_COLORS[index % len(_LAYER_COLORS)], width=2.4),
                    marker=dict(size=8),
                    hovertemplate="L" + str(layer) + "<br>active SAE features=%{x}<br>macro-F1=%{y:.3f}<extra></extra>",
                )
            )
        figure.update_xaxes(title="Retained SAE latents", type="category")
        figure.update_yaxes(title="Held-out macro-F1", range=[0, 1], tickformat=".1f")
        destination = output_dir / f"figure_sae_feature_budget_{_safe_label(model)}.html"
        _write_figure(figure, destination, f"SAE feature-budget sensitivity: {model}")
        paths.extend((str(destination), str(destination.with_suffix(".pdf"))))
    return paths


def _domain_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    paths = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["layer"]), []).append(row)
    for (model, layer), group in sorted(groups.items()):
        domains = sorted({row["domain"] for row in group})
        experts = sorted({(row["expert"], row["expert_label"]) for row in group})
        views = sorted({str(row.get("view", "Observed router")) for row in group})
        figure = make_subplots(
            rows=1,
            cols=len(views),
            subplot_titles=tuple(views),
            horizontal_spacing=0.12,
        )
        for column, view in enumerate(views, start=1):
            view_rows = [row for row in group if str(row.get("view", "Observed router")) == view]
            matrix = []
            for _, label in experts:
                values = {
                    row["domain"]: row["routing_share"]
                    for row in view_rows
                    if row["expert_label"] == label
                }
                matrix.append([values.get(domain, 0.0) for domain in domains])
            figure.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=domains,
                    y=[label for _, label in experts],
                    colorscale=_BLUE_HEATMAP,
                    zmin=0,
                    zmax=1,
                    showscale=column == len(views),
                    colorbar=dict(
                    title=dict(
                        text="Activation rate",
                        side="right",
                        font=dict(size=24, family="Arial, sans-serif", weight=600),
                    ),
                        tickformat=".0%",
                        tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
                        len=0.76,
                        thickness=18,
                        x=1.015,
                    tickfont=dict(size=22, family="Arial, sans-serif", weight=600),
                    ) if column == len(views) else None,
                    hovertemplate=(
                        f"view={view}<br>expert=%{{y}}<br>domain=%{{x}}"
                        "<br>activation rate=%{z:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.update_xaxes(title="Dataset domain", tickangle=-35, automargin=True, row=1, col=column)
            figure.update_yaxes(
                title="Expert" if column == 1 else None,
                autorange="reversed",
                automargin=True,
                row=1,
                col=column,
            )
        destination = output_dir / f"domain_routing_{_safe_label(model)}_layer_{layer:02d}.html"
        _write_figure(
            figure,
            destination,
            f"Observed vs. SAE-predicted routing by domain (L{layer})",
            height=max(680, 150 + 66 * len(experts)),
            has_subplots=len(views) > 1,
        )
        paths.append(str(destination))
        paths.append(str(destination.with_suffix(".pdf")))
    return paths


def _feature_health_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    paths: list[str] = []
    for model, group in sorted(_by_model(rows).items()):
        figure = go.Figure()
        group.sort(key=lambda row: row["layer"])
        figure.add_trace(
            go.Bar(
                x=[f"L{row['layer']}" for row in group],
                y=[row["dead_feature_fraction"] for row in group],
                marker_color="#287F8C",
                showlegend=False,
                hovertemplate="%{x}<br>dead features=%{y:.3f}<extra></extra>",
            )
        )
        figure.update_yaxes(title="Dead SAE feature fraction", range=[0, 1], tickformat=".0%")
        figure.update_xaxes(title="Transformer layer", type="category")
        destination = output_dir / f"figure_sae_feature_health_{_safe_label(model)}.html"
        _write_figure(figure, destination, f"SAE feature health: {model}", height=560)
        paths.extend((str(destination), str(destination.with_suffix(".pdf"))))
    return paths


def _router_distribution_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Appendix histograms from measured router probabilities, grouped by layer."""

    paths: list[str] = []
    by_model = _by_model(rows)
    metrics = (
        ("top1_weight", "Top-1 routing weight"),
        ("top1_top2_margin", "Top-1 vs. top-2 margin"),
        ("normalized_entropy", "Normalised router entropy"),
    )
    for model, model_rows in sorted(by_model.items()):
        figure = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=tuple(label for _, label in metrics),
            horizontal_spacing=0.10,
        )
        layers = sorted({int(row["layer"]) for row in model_rows})
        for column, (metric, _) in enumerate(metrics, start=1):
            for index, layer in enumerate(layers):
                group = sorted(
                    (row for row in model_rows if row["metric"] == metric and int(row["layer"]) == layer),
                    key=lambda row: float(row["bin_left"]),
                )
                if not group:
                    continue
                x = [(float(row["bin_left"]) + float(row["bin_right"])) / 2 for row in group]
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=[float(row["density"]) for row in group],
                        mode="lines",
                        name=f"L{layer}",
                        legendgroup=f"L{layer}",
                        showlegend=column == 1,
                        line=dict(color=_LAYER_COLORS[index % len(_LAYER_COLORS)], width=2.4),
                        hovertemplate="L" + str(layer) + "<br>value=%{x:.3f}<br>token share=%{y:.3f}<extra></extra>",
                    ),
                    row=1,
                    col=column,
                )
            figure.update_xaxes(title="Probability", range=[0, 1], row=1, col=column)
            figure.update_yaxes(title="Token share", rangemode="tozero", row=1, col=column)
        destination = output_dir / f"figure_router_distributions_{_safe_label(model)}.html"
        _write_figure(
            figure,
            destination,
            f"Router distributions: {model}",
            height=720,
            has_subplots=True,
        )
        paths.extend((str(destination), str(destination.with_suffix(".pdf"))))
    return paths


def _model_summary_figures(
    routing_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    health_rows: list[dict[str, Any]],
    domain_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Render one self-contained, readable appendix overview for each model."""

    paths: list[str] = []
    routing_by_model = _by_model(routing_rows)
    budget_by_model = _by_model(budget_rows)
    health_by_model = _by_model(health_rows)
    domain_by_model = _by_model(domain_rows)
    for model, model_routing in sorted(routing_by_model.items()):
        layers = sorted({int(row["layer"]) for row in model_routing})
        primary_layer = layers[len(layers) // 2]
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Router prediction (macro-F1)",
                f"SAE feature budget (L{primary_layer})",
                "Dead SAE features",
                f"Expert allocation by domain (L{primary_layer})",
            ),
            horizontal_spacing=0.12,
            vertical_spacing=0.07,
        )

        for method in _METHOD_LABELS.values():
            group = sorted(
                (row for row in model_routing if row["method"] == method),
                key=lambda row: row["layer"],
            )
            if not group:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[f"L{row['layer']}" for row in group],
                    y=[row["macro_f1"] for row in group],
                    mode="lines+markers",
                    name=method,
                    showlegend=True,
                    line=dict(color=_METHOD_COLORS[method], width=3 if method == "SAE predictor" else 2),
                    marker=dict(size=7),
                    hovertemplate="%{fullData.name}<br>%{x}<br>macro-F1=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        figure.update_xaxes(title=None, type="category", row=1, col=1)
        figure.update_yaxes(title="Macro-F1", range=[0, 1], tickformat=".1f", row=1, col=1)
        budget_group = sorted(
            (row for row in budget_by_model.get(model, []) if int(row["layer"]) == primary_layer),
            key=lambda row: row["active_features"],
        )
        figure.add_trace(
            go.Scatter(
                x=[str(row["active_features"]) for row in budget_group],
                y=[row["macro_f1"] for row in budget_group],
                mode="lines+markers",
                name="SAE budget",
                showlegend=False,
                line=dict(color="#063F59", width=3),
                marker=dict(size=7),
                hovertemplate="retained SAE latents=%{x}<br>macro-F1=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.update_xaxes(title=None, type="category", row=1, col=2)
        figure.update_yaxes(title="Macro-F1", range=[0, 1], tickformat=".1f", row=1, col=2)

        health_group = sorted(health_by_model.get(model, []), key=lambda row: row["layer"])
        figure.add_trace(
            go.Bar(
                x=[f"L{row['layer']}" for row in health_group],
                y=[row["dead_feature_fraction"] for row in health_group],
                name="Dead SAE features",
                showlegend=False,
                marker_color="#287F8C",
                hovertemplate="%{x}<br>dead features=%{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.update_xaxes(title="Layer", type="category", row=2, col=1)
        figure.update_yaxes(title="Fraction", range=[0, 1], tickformat=".0%", row=2, col=1)

        domain_group = [
            row
            for row in domain_by_model.get(model, [])
            if int(row["layer"]) == primary_layer and row.get("view", "Observed router") == "Observed router"
        ]
        if not domain_group:
            domain_group = [
                row for row in domain_by_model.get(model, []) if int(row["layer"]) == primary_layer
            ]
        domains = sorted({str(row["domain"]) for row in domain_group})
        experts = sorted({(int(row["expert"]), str(row["expert_label"])) for row in domain_group})
        matrix = []
        for _, label in experts:
            values = {str(row["domain"]): float(row["routing_share"]) for row in domain_group if row["expert_label"] == label}
            matrix.append([values.get(domain, 0.0) for domain in domains])
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=domains,
                y=[label for _, label in experts],
                colorscale=_BLUE_HEATMAP,
                zmin=0,
                zmax=1,
                colorbar=dict(
                    title=dict(
                        text="Activation rate",
                        side="right",
                        font=dict(size=24, family="Arial, sans-serif", weight=600),
                    ),
                    tickformat=".0%",
                    thickness=18,
                    len=0.32,
                    x=1.015,
                    tickfont=dict(size=22, family="Arial, sans-serif", weight=600),
                ),
                hovertemplate="expert=%{y}<br>domain=%{x}<br>share=%{z:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        figure.update_xaxes(title="Dataset domain", tickangle=-35, automargin=True, row=2, col=2)
        figure.update_yaxes(title="Expert", autorange="reversed", automargin=True, row=2, col=2)

        figure.update_layout(
            template="plotly_white",
            title=dict(
                text=f"<b>{model}</b>", x=0.02, xanchor="left", y=0.99,
                font=dict(size=28, color=_INK, family="Arial, sans-serif"),
            ),
            font=dict(family="Arial, sans-serif", size=24, color=_INK),
            paper_bgcolor=_PAPER_BG,
            plot_bgcolor=_PAPER_BG,
            width=1280,
            height=1040,
            margin=dict(l=92, r=122, t=66, b=68),
            legend=dict(
                orientation="h", yanchor="bottom", yref="container", y=0.91, xanchor="center", x=0.5,
                font=dict(size=23, family="Arial, sans-serif", weight=600),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        figure.update_xaxes(
            showline=True, linecolor="#8BA4AA", gridcolor=_GRID, zeroline=False,
            title_font=dict(size=24, family="Arial, sans-serif", weight=600),
            tickfont=dict(size=22, family="Arial, sans-serif", weight=600), title_standoff=10,
        )
        figure.update_yaxes(
            showline=True, linecolor="#8BA4AA", gridcolor=_GRID, zeroline=False,
            title_font=dict(size=24, family="Arial, sans-serif", weight=600),
            tickfont=dict(size=22, family="Arial, sans-serif", weight=600), title_standoff=10,
        )
        figure.update_layout(
            yaxis=dict(domain=[0.50, 0.94]),
            yaxis2=dict(domain=[0.50, 0.94]),
            yaxis3=dict(domain=[0.08, 0.37]),
            yaxis4=dict(domain=[0.08, 0.37]),
        )
        for index, annotation in enumerate(figure.layout.annotations or ()):
            annotation.y = 0.97 if index < 2 else 0.40
            annotation.update(font=dict(size=26, family="Arial, sans-serif", weight=600, color=_INK))
        destination = output_dir / f"figure_router_summary_{_safe_label(model)}.html"
        save_plotly_figure(figure, destination, format="html")
        save_plotly_figure(figure, destination.with_suffix(".pdf"), format="pdf")
        paths.extend((str(destination), str(destination.with_suffix(".pdf"))))
    return paths


def render_routerinterp_report(summary_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    """Render tables and direct router-evidence figures from completed analyses."""

    output_dir.mkdir(parents=True, exist_ok=True)
    input_summary_paths = [str(path) for path in summary_paths]
    routing_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for path in summary_paths:
        routing, budget, health, domain, distributions = _collect_summary(path)
        routing_rows.extend(routing)
        budget_rows.extend(budget)
        health_rows.extend(health)
        domain_rows.extend(domain)
        distribution_rows.extend(distributions)
    if not routing_rows:
        raise ValueError("No layer results were found in the supplied RouterInterp summaries.")

    routing_fields = _fieldnames(routing_rows)
    budget_fields = _fieldnames(budget_rows)
    health_fields = _fieldnames(health_rows)
    domain_fields = _fieldnames(domain_rows)
    distribution_fields = _fieldnames(distribution_rows)
    _write_csv(output_dir / "routing_predictor_metrics.csv", routing_rows, routing_fields)
    _write_csv(output_dir / "sae_feature_budget_metrics.csv", budget_rows, budget_fields)
    _write_csv(output_dir / "sae_feature_health.csv", health_rows, health_fields)
    _write_csv(output_dir / "expert_domain_routing.csv", domain_rows, domain_fields)
    if distribution_rows:
        _write_csv(output_dir / "router_probability_distributions.csv", distribution_rows, distribution_fields)
    _write_predictor_table(routing_rows, output_dir / "table_direct_routing_predictors.tex")
    _write_main_predictor_table(routing_rows, output_dir / "table_direct_routing_predictors_mean.tex")
    predictor_paths = _predictor_figures(routing_rows, output_dir)
    budget_paths = _budget_figures(budget_rows, output_dir)
    health_paths = _feature_health_figures(health_rows, output_dir)
    domain_paths = _domain_figures(domain_rows, output_dir)
    distribution_paths = _router_distribution_figures(distribution_rows, output_dir) if distribution_rows else []
    model_summary_paths = _model_summary_figures(
        routing_rows,
        budget_rows,
        health_rows,
        domain_rows,
        output_dir,
    )
    manifest = {
        "summary_paths": input_summary_paths,
        "output_dir": str(output_dir),
        "tables": [
            "routing_predictor_metrics.csv",
            "sae_feature_budget_metrics.csv",
            "sae_feature_health.csv",
            "expert_domain_routing.csv",
            *( ["router_probability_distributions.csv"] if distribution_rows else [] ),
            "table_direct_routing_predictors.tex",
            "table_direct_routing_predictors_mean.tex",
        ],
        "figures": [
            *predictor_paths,
            *budget_paths,
            *health_paths,
            *domain_paths,
            *distribution_paths,
            *model_summary_paths,
        ],
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
