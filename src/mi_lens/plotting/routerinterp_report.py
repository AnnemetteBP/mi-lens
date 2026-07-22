"""Publication-oriented reports for direct RouterInterp routing evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

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
_DOMAIN_COLORS = [
    "#063F59", "#287F8C", "#5CA7B1", "#8FC7CE", "#4B7586", "#6E98A3",
    "#9C7B59", "#9B5B6A", "#6C7B45", "#765C91", "#BA6A36", "#537E6A",
]
_EXPERT_COLORS = [
    "#063F59", "#287F8C", "#5CA7B1", "#4B7586", "#9C7B59", "#9B5B6A",
    "#6C7B45", "#765C91", "#BA6A36", "#537E6A",
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
    width: int = 1280,
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
        width=width,
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


def _discrete_colorscale(colors: list[str]) -> list[list[object]]:
    """Return a step scale so integer expert IDs remain categorical."""

    if not colors:
        raise ValueError("A categorical colourscale requires at least one colour.")
    return [
        point
        for index, color in enumerate(colors)
        for point in ([index / len(colors), color], [(index + 1) / len(colors), color])
    ]


def _load_router_neighbourhood(
    summary_path: Path,
    *,
    max_tokens_per_domain: int = 250,
) -> tuple[str, int, list[str], torch.Tensor, list[str], torch.Tensor]:
    """Load a balanced held-out router-input sample for one projection figure."""

    if max_tokens_per_domain < 2:
        raise ValueError("max_tokens_per_domain must be at least two.")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("format") != "mi_lens.routerinterp.analysis.v1":
        raise ValueError(f"{summary_path} is not a RouterInterp analysis summary.")
    eval_root = Path(str(payload.get("eval_artifacts_path", "")))
    manifest_path = eval_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing held-out router capture manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = sorted(int(layer) for layer in payload.get("layers", {}))
    if not layers:
        raise ValueError(f"{summary_path} contains no analysed layers.")
    layer = layers[len(layers) // 2]
    expert_labels = [str(label) for label in payload.get("expert_labels", [])]
    if not expert_labels:
        raise ValueError(f"{summary_path} does not record expert labels.")

    per_domain_counts: dict[str, int] = {}
    inputs: list[torch.Tensor] = []
    domains: list[str] = []
    selected: list[torch.Tensor] = []
    for record in manifest.get("prompts", []):
        domain = str(record.get("domain", "")).strip()
        if not domain or domain.lower() == "unknown":
            raise ValueError("Router neighbourhood figures require an explicit dataset-domain label for every prompt.")
        remaining = max_tokens_per_domain - per_domain_counts.get(domain, 0)
        if remaining <= 0:
            continue
        capture_path = eval_root / str(record["path"])
        if not capture_path.is_file():
            raise FileNotFoundError(f"Missing held-out router capture: {capture_path}")
        capture = torch.load(capture_path, map_location="cpu", weights_only=True)
        layer_payload = capture.get("layers", {}).get(str(layer))
        if layer_payload is None:
            raise KeyError(f"{capture_path} has no router capture for layer {layer}.")
        router_input = layer_payload.get("router_input")
        selected_experts = layer_payload.get("selected_experts")
        if not isinstance(router_input, torch.Tensor) or not isinstance(selected_experts, torch.Tensor):
            raise TypeError(f"{capture_path} has malformed router capture tensors.")
        if router_input.ndim != 2 or selected_experts.ndim != 2 or router_input.shape[0] != selected_experts.shape[0]:
            raise ValueError(f"{capture_path} router inputs and selected experts have incompatible shapes.")
        take = min(remaining, int(router_input.shape[0]))
        if take == 0:
            continue
        ids = selected_experts[:take, 0].to(dtype=torch.long)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(expert_labels)):
            raise ValueError(f"{capture_path} contains an expert ID outside the recorded expert-label range.")
        values = router_input[:take].to(dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError(f"{capture_path} contains non-finite router inputs.")
        inputs.append(values)
        selected.append(ids)
        domains.extend([domain] * take)
        per_domain_counts[domain] = per_domain_counts.get(domain, 0) + take
    if len(inputs) < 2:
        raise ValueError("Router neighbourhood figures require at least two held-out captured prompts.")
    router_inputs = torch.cat(inputs, dim=0)
    expert_ids = torch.cat(selected, dim=0)
    if router_inputs.shape[0] < 3:
        raise ValueError("Router neighbourhood figures require at least three captured tokens.")
    return _model_label(summary_path, payload), layer, expert_labels, router_inputs, domains, expert_ids


def _pca_coordinates(router_inputs: torch.Tensor) -> torch.Tensor:
    """Compute a checked two-dimensional PCA projection of router inputs."""

    if router_inputs.ndim != 2 or min(router_inputs.shape) < 2:
        raise ValueError("Two-dimensional PCA requires at least two tokens and two router-input dimensions.")
    centered = router_inputs - router_inputs.mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(centered, q=2, center=False, niter=4)
    coordinates = centered @ vectors[:, :2]
    if not torch.isfinite(coordinates).all():
        raise ValueError("Router-input PCA produced non-finite coordinates.")
    return coordinates


def _knn_region_labels(coordinates: torch.Tensor, expert_ids: torch.Tensor, n_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify a PCA-plane grid by actual selected-expert nearest neighbours."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("kNN decision regions require two-dimensional coordinates.")
    if expert_ids.shape != (coordinates.shape[0],):
        raise ValueError("kNN decision regions require one selected expert ID per coordinate.")
    lower = coordinates.min(dim=0).values
    upper = coordinates.max(dim=0).values
    span = (upper - lower).clamp_min(1e-5)
    lower = lower - 0.08 * span
    upper = upper + 0.08 * span
    axis_x = torch.linspace(float(lower[0]), float(upper[0]), 90)
    axis_y = torch.linspace(float(lower[1]), float(upper[1]), 90)
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    query = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    k = min(15, int(coordinates.shape[0]))
    labels: list[torch.Tensor] = []
    for start in range(0, int(query.shape[0]), 512):
        distances = torch.cdist(query[start : start + 512], coordinates)
        neighbours = expert_ids[distances.topk(k=k, largest=False).indices]
        votes = torch.zeros((neighbours.shape[0], n_experts), dtype=torch.int64)
        votes.scatter_add_(1, neighbours, torch.ones_like(neighbours, dtype=torch.int64))
        labels.append(votes.argmax(dim=1))
    return axis_x, axis_y, torch.cat(labels).reshape(grid_y.shape)


def _router_neighbourhood_figures(summary_paths: list[Path], output_dir: Path) -> list[str]:
    """Show observed domain structure and selected-expert regions in PCA space."""

    paths: list[str] = []
    for summary_path in summary_paths:
        model, layer, expert_labels, router_inputs, domains, expert_ids = _load_router_neighbourhood(summary_path)
        coordinates = _pca_coordinates(router_inputs)
        axis_x, axis_y, region_labels = _knn_region_labels(coordinates, expert_ids, len(expert_labels))
        domain_names = sorted(set(domains))
        if len(domain_names) > len(_DOMAIN_COLORS):
            raise ValueError("Add domain colours before plotting more than twelve dataset domains.")
        domain_ids = [domain_names.index(domain) for domain in domains]
        domain_figure = go.Figure()
        domain_figure.add_trace(
            go.Scatter(
                x=coordinates[:, 0].tolist(),
                y=coordinates[:, 1].tolist(),
                customdata=domains,
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=5,
                    color=domain_ids,
                    colorscale=_discrete_colorscale(_DOMAIN_COLORS[: len(domain_names)]),
                    cmin=0,
                    cmax=max(1, len(domain_names) - 1),
                    showscale=True,
                    opacity=0.70,
                    colorbar=dict(
                        title=dict(text="Dataset domain", side="right"),
                        tickmode="array",
                        tickvals=list(range(len(domain_names))),
                        ticktext=domain_names,
                        thickness=16,
                    ),
                ),
                hovertemplate="domain=%{customdata}<br>PC 1=%{x:.2f}<br>PC 2=%{y:.2f}<extra></extra>",
            )
        )
        domain_figure.update_xaxes(title="Router-input PC 1")
        domain_figure.update_yaxes(title="Router-input PC 2")
        domain_destination = output_dir / f"figure_router_domain_neighbourhood_{_safe_label(model)}_layer_{layer:02d}.html"
        _write_figure(
            domain_figure,
            domain_destination,
            f"Router-input neighbourhood by dataset domain: {model} (L{layer})",
            height=700,
            width=980,
        )
        paths.extend((str(domain_destination), str(domain_destination.with_suffix(".pdf"))))

        colours = _EXPERT_COLORS[: len(expert_labels)]
        if len(colours) < len(expert_labels):
            raise ValueError("Add expert colours before plotting a model with more than ten experts.")
        expert_figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Observed selected expert", "15-NN selected-expert regions"),
            horizontal_spacing=0.12,
        )
        expert_figure.add_trace(
            go.Scatter(
                x=coordinates[:, 0].tolist(),
                y=coordinates[:, 1].tolist(),
                customdata=[expert_labels[int(expert)] for expert in expert_ids.tolist()],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=5,
                    color=expert_ids.tolist(),
                    colorscale=_discrete_colorscale(colours),
                    cmin=0,
                    cmax=max(1, len(expert_labels) - 1),
                    showscale=False,
                    opacity=0.70,
                ),
                hovertemplate="observed expert=%{customdata}<br>PC 1=%{x:.2f}<br>PC 2=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        expert_figure.add_trace(
            go.Heatmap(
                x=axis_x.tolist(),
                y=axis_y.tolist(),
                z=region_labels.tolist(),
                zmin=0,
                zmax=max(1, len(expert_labels) - 1),
                colorscale=_discrete_colorscale(colours),
                showscale=True,
                opacity=0.48,
                colorbar=dict(
                    title=dict(text="15-NN expert", side="right"),
                    tickmode="array",
                    tickvals=list(range(len(expert_labels))),
                    ticktext=expert_labels,
                    thickness=16,
                ),
                hovertemplate="PC 1=%{x:.2f}<br>PC 2=%{y:.2f}<br>15-NN expert=%{z}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        for expert_id, label in enumerate(expert_labels):
            mask = expert_ids == expert_id
            if not bool(mask.any()):
                continue
            expert_figure.add_trace(
                go.Scatter(
                    x=coordinates[mask, 0].tolist(),
                    y=coordinates[mask, 1].tolist(),
                    mode="markers",
                    name=label,
                    showlegend=False,
                    marker=dict(size=3.5, color=colours[expert_id], opacity=0.88, line=dict(width=0.2, color="#FFFFFF")),
                    hovertemplate=f"observed expert={label}<br>PC 1=%{{x:.2f}}<br>PC 2=%{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        for column in range(1, 3):
            expert_figure.update_xaxes(title="Router-input PC 1", row=1, col=column)
            expert_figure.update_yaxes(title="Router-input PC 2", row=1, col=column)
        expert_destination = output_dir / f"figure_router_expert_regions_{_safe_label(model)}_layer_{layer:02d}.html"
        _write_figure(
            expert_figure,
            expert_destination,
            f"Observed expert routing and 15-NN regions: {model} (L{layer})",
            height=700,
            has_subplots=True,
            width=1220,
        )
        paths.extend((str(expert_destination), str(expert_destination.with_suffix(".pdf"))))
    return paths


def _feature_domain_profile_figures(summary_paths: list[Path], output_dir: Path) -> list[str]:
    """Render observed broad-domain profiles for top routing-useful SAE latents."""

    paths: list[str] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("format") != "mi_lens.routerinterp.analysis.v1":
            raise ValueError(f"{summary_path} is not a RouterInterp analysis summary.")
        layers = sorted(int(layer) for layer in payload.get("layers", {}))
        if not layers:
            continue
        layer = layers[len(layers) // 2]
        layer_result = payload["layers"][str(layer)]
        profile_path = Path(str(layer_result.get("top_rho_domain_profile_artifact", "")))
        if not profile_path.is_file():
            raise FileNotFoundError(f"Missing top-rho feature-domain profile artifact: {profile_path}")
        profile_rows = json.loads(profile_path.read_text(encoding="utf-8"))
        if not profile_rows:
            continue
        model = _model_label(summary_path, payload)
        domains = sorted({str(row["domain"]) for row in profile_rows})
        if any(not domain or domain.lower() == "unknown" for domain in domains):
            raise ValueError("Top-rho feature-domain figures require explicit dataset-domain labels.")
        feature_groups: dict[tuple[int, str, int, int], list[dict[str, Any]]] = {}
        for row in profile_rows:
            share = _validate_metric(
                "activation_mass_share",
                row["activation_mass_share"],
                context=f"{model}, layer {layer}, top-rho feature profile",
            )
            entropy = _validate_metric(
                "normalized_domain_entropy",
                row["normalized_domain_entropy"],
                context=f"{model}, layer {layer}, top-rho feature profile",
            )
            if share < 0.0 or share > 1.0 or entropy < 0.0 or entropy > 1.0:
                raise ValueError("Top-rho feature-domain metrics must be in [0, 1].")
            key = (int(row["expert"]), str(row["expert_label"]), int(row["rho_rank"]), int(row["feature_id"]))
            feature_groups.setdefault(key, []).append(row)
        ordered = sorted(feature_groups.items(), key=lambda item: item[0])
        y_labels = [f"{label}: f{feature_id}" for (_, label, _, feature_id), _ in ordered]
        matrix: list[list[float]] = []
        entropies: dict[str, list[float]] = {}
        for (_, label, _, _), rows in ordered:
            by_domain = {str(row["domain"]): float(row["activation_mass_share"]) for row in rows}
            values = [by_domain.get(domain, 0.0) for domain in domains]
            if not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Top-rho feature-domain activation shares must sum to one per feature.")
            matrix.append(values)
            entropies.setdefault(label, []).append(float(rows[0]["normalized_domain_entropy"]))
        figure = go.Figure(
            go.Heatmap(
                z=matrix,
                x=domains,
                y=y_labels,
                colorscale=_BLUE_HEATMAP,
                zmin=0,
                zmax=1,
                colorbar=dict(title=dict(text="Activation mass share", side="right"), tickformat=".0%", thickness=16),
                hovertemplate="feature=%{y}<br>domain=%{x}<br>activation share=%{z:.3f}<extra></extra>",
            )
        )
        figure.update_xaxes(title="Dataset domain", tickangle=-30)
        figure.update_yaxes(title="Top-rho SAE feature", autorange="reversed")
        destination = output_dir / f"figure_top_rho_feature_domain_profiles_{_safe_label(model)}_layer_{layer:02d}.html"
        _write_figure(
            figure,
            destination,
            f"Top-rho SAE feature profiles by dataset domain: {model} (L{layer})",
            height=max(720, 220 + 20 * len(y_labels)),
            width=1240,
        )
        paths.extend((str(destination), str(destination.with_suffix(".pdf"))))

        labels = list(entropies)
        diversity = [sum(entropies[label]) / len(entropies[label]) for label in labels]
        diversity_figure = go.Figure(
            go.Bar(
                x=labels,
                y=diversity,
                marker_color=[_EXPERT_COLORS[index % len(_EXPERT_COLORS)] for index in range(len(labels))],
                hovertemplate="expert=%{x}<br>mean feature-domain entropy=%{y:.3f}<extra></extra>",
            )
        )
        diversity_figure.update_xaxes(title="Expert")
        diversity_figure.update_yaxes(title="Mean normalized domain entropy", range=[0, 1], tickformat=".1f")
        diversity_destination = output_dir / f"figure_top_rho_feature_domain_diversity_{_safe_label(model)}_layer_{layer:02d}.html"
        _write_figure(
            diversity_figure,
            diversity_destination,
            f"Broad-domain diversity of top-rho SAE features: {model} (L{layer})",
            height=600,
            width=980,
        )
        paths.extend((str(diversity_destination), str(diversity_destination.with_suffix(".pdf"))))
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
    neighbourhood_paths = _router_neighbourhood_figures(summary_paths, output_dir)
    feature_profile_paths = _feature_domain_profile_figures(summary_paths, output_dir)
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
            *neighbourhood_paths,
            *feature_profile_paths,
            *model_summary_paths,
        ],
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
