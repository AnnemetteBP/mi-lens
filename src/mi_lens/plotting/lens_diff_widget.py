from __future__ import annotations

import base64
import html

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
import torch


def build_lens_comparison_widget(
    model,
    lens,
    tok,
    prompt: str,
    *,
    tuned_lens=None,
    layer_stride: int = 1,
    last_n_positions: int = 12,
    top_k: int = 8,
    correct_k: int = 5,
):
    axis_token_limit = 14
    cell_token_limit = 10
    figure_width = 1200
    figure_height = 690
    colorscale_options = [
        "Auto",
        "Viridis",
        "Cividis",
        "Cividis_r",
        "Magma",
        "Magma_r",
        "Plasma",
        "Inferno",
        "Turbo",
        "Blues",
        "Greens",
        "YlOrRd",
        "RdBu",
    ]
    marker_color_options = [
        "black",
        "gray",
        "deeppink",
        "crimson",
        "dodgerblue",
        "seagreen",
        "darkorange",
        "goldenrod",
        "mediumpurple",
        "teal",
        "brown",
    ]
    lens_names = {
        "jlens": "J-lens",
        "logit": "Logit lens",
    }
    lens_short_names = {
        "jlens": "JL",
        "logit": "LL",
    }
    if tuned_lens is not None:
        lens_names["tuned"] = "Tuned lens"
        lens_short_names["tuned"] = "TL"

    metrics = {
        "jaccard_pair": "Jaccard(left top-k, right top-k)",
        "jaccard_left_final": "Jaccard(left top-k, final top-k)",
        "jaccard_right_final": "Jaccard(right top-k, final top-k)",
        "rank_final_top_left": "Rank of final top-1 under left lens",
        "rank_final_top_right": "Rank of final top-1 under right lens",
        "kl_left_final": "KL(left || final)",
        "kl_right_final": "KL(right || final)",
        "tv_left_final": "TV(left, final)",
        "tv_right_final": "TV(right, final)",
        "js_left_final": "JS(left, final)",
        "js_right_final": "JS(right, final)",
        "gt_prob_left": "Prob(final top-1 | left)",
        "gt_prob_right": "Prob(final top-1 | right)",
        "gt_prob_diff_left_minus_right": "Prob(final top-1): left - right",
    }

    def decode_token(token_id: int) -> str:
        return tok.decode(
            [int(token_id)], clean_up_tokenization_spaces=False
        ).replace("\n", "\\n")

    def truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."

    def topk_ids(logits: torch.Tensor, k: int) -> list[int]:
        return logits.topk(k).indices.tolist()

    def jaccard(ids_a: list[int], ids_b: list[int]) -> float:
        a, b = set(ids_a), set(ids_b)
        return len(a & b) / len(a | b) if (a or b) else 1.0

    def rank_of_token(logits: torch.Tensor, token_id: int) -> int:
        return int(
            (logits.argsort(descending=True) == token_id).nonzero()[0].item() + 1
        )

    def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=0)

    def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        return float((p * (p.log() - q.log())).sum().item())

    def tv_dist(p: torch.Tensor, q: torch.Tensor) -> float:
        return float(0.5 * torch.abs(p - q).sum().item())

    def js_div(p: torch.Tensor, q: torch.Tensor) -> float:
        m = 0.5 * (p + q)
        return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

    def parse_optional_int(raw: str, default: int) -> int:
        raw = raw.strip()
        if not raw:
            return default
        return int(raw)

    def compact_layout(min_width: str = "120px") -> widgets.Layout:
        return widgets.Layout(width="auto", min_width=min_width, margin="0")

    def metric_value(
        metric: str,
        left: torch.Tensor,
        right: torch.Tensor,
        final: torch.Tensor,
        compare_top_k: int,
    ) -> float:
        left_ids = topk_ids(left, compare_top_k)
        right_ids = topk_ids(right, compare_top_k)
        final_ids = topk_ids(final, compare_top_k)
        final_top = final_ids[0]

        p_left = softmax_probs(left)
        p_right = softmax_probs(right)
        p_final = softmax_probs(final)

        if metric == "jaccard_pair":
            return jaccard(left_ids, right_ids)
        if metric == "jaccard_left_final":
            return jaccard(left_ids, final_ids)
        if metric == "jaccard_right_final":
            return jaccard(right_ids, final_ids)
        if metric == "rank_final_top_left":
            return rank_of_token(left, final_top)
        if metric == "rank_final_top_right":
            return rank_of_token(right, final_top)
        if metric == "kl_left_final":
            return kl_div(p_left, p_final)
        if metric == "kl_right_final":
            return kl_div(p_right, p_final)
        if metric == "tv_left_final":
            return tv_dist(p_left, p_final)
        if metric == "tv_right_final":
            return tv_dist(p_right, p_final)
        if metric == "js_left_final":
            return js_div(p_left, p_final)
        if metric == "js_right_final":
            return js_div(p_right, p_final)
        if metric == "gt_prob_left":
            return float(p_left[final_top].item())
        if metric == "gt_prob_right":
            return float(p_right[final_top].item())
        if metric == "gt_prob_diff_left_minus_right":
            return float((p_left[final_top] - p_right[final_top]).item())
        raise ValueError(metric)

    def metric_style(metric: str, z: np.ndarray) -> dict[str, object]:
        if metric in {
            "jaccard_pair",
            "jaccard_left_final",
            "jaccard_right_final",
            "gt_prob_left",
            "gt_prob_right",
        }:
            return {"colorscale": "Viridis", "zmin": 0.0, "zmax": 1.0}
        if metric == "gt_prob_diff_left_minus_right":
            lim = max(abs(float(z.min())), abs(float(z.max())), 1e-6)
            return {"colorscale": "RdBu", "zmin": -lim, "zmax": lim}
        if metric.startswith("rank_"):
            return {
                "colorscale": "Magma_r",
                "zmin": float(z.min()),
                "zmax": float(z.max()),
            }
        return {
            "colorscale": "Cividis_r",
            "zmin": float(z.min()),
            "zmax": float(z.max()),
        }

    def metric_label(metric: str, left_key: str, right_key: str) -> str:
        left_name = lens_names[left_key]
        right_name = lens_names[right_key]
        labels = {
            "jaccard_pair": f"Jaccard({left_name} top-k, {right_name} top-k)",
            "jaccard_left_final": f"Jaccard({left_name} top-k, Final top-k)",
            "jaccard_right_final": f"Jaccard({right_name} top-k, Final top-k)",
            "rank_final_top_left": f"Rank of final top-1 under {left_name}",
            "rank_final_top_right": f"Rank of final top-1 under {right_name}",
            "kl_left_final": f"KL({left_name} || Final)",
            "kl_right_final": f"KL({right_name} || Final)",
            "tv_left_final": f"TV({left_name}, Final)",
            "tv_right_final": f"TV({right_name}, Final)",
            "js_left_final": f"JS({left_name}, Final)",
            "js_right_final": f"JS({right_name}, Final)",
            "gt_prob_left": f"Prob(final top-1 | {left_name})",
            "gt_prob_right": f"Prob(final top-1 | {right_name})",
            "gt_prob_diff_left_minus_right": f"Prob(final top-1): {left_name} - {right_name}",
        }
        return labels[metric]

    def compute_tuned_logits(
        input_ids: torch.Tensor,
        positions: list[int],
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        if tuned_lens is None:
            return {}
        hf_model = getattr(model, "_hf_model", None)
        if hf_model is None:
            raise ValueError(
                "Tuned-lens comparison requires a jlens HFLensModel built via jlens.from_hf(...)."
            )
        with torch.no_grad():
            outputs = hf_model(
                input_ids=input_ids.to(model.input_device),
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states[:-1]
        tuned_device = next(tuned_lens.parameters()).device
        tuned_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            tuned_idx = layer + 1
            if tuned_idx >= len(hidden_states):
                raise ValueError(
                    f"Tuned-lens hidden-state index {tuned_idx} is out of range for layer L{layer}."
                )
            hidden = hidden_states[tuned_idx][0]
            selected = hidden[list(positions)].to(tuned_device)
            tuned_logits[layer] = tuned_lens(selected, idx=tuned_idx).float().cpu()
        return tuned_logits

    def compute_prompt_state(
        prompt_text: str,
        stride: int,
        n_positions: int,
        compare_top_k: int,
        marker_top_k: int,
        start_token_id_raw: str,
        end_token_id_raw: str,
    ) -> dict[str, object]:
        input_ids_full = model.encode(prompt_text)
        token_ids = input_ids_full[0].tolist()
        seq_len = len(token_ids)
        if seq_len < 2:
            raise ValueError("Prompt must tokenize to at least 2 tokens.")

        valid_max_position = seq_len - 2
        use_explicit_window = bool(start_token_id_raw.strip() or end_token_id_raw.strip())

        if use_explicit_window:
            start_token_id = parse_optional_int(start_token_id_raw, 0)
            end_token_id = parse_optional_int(
                end_token_id_raw, min(13, valid_max_position)
            )
            start_token_id = max(0, min(start_token_id, valid_max_position))
            end_token_id = max(0, min(end_token_id, valid_max_position))
            if end_token_id < start_token_id:
                raise ValueError(
                    "End token id must be greater than or equal to start token id."
                )
            positions = list(range(start_token_id, end_token_id + 1))
        else:
            n_positions = max(1, int(n_positions))
            positions = list(range(max(0, (seq_len - 1) - n_positions), seq_len - 1))

        if not positions:
            raise ValueError("Prompt did not produce any prediction positions to display.")

        stride = max(1, int(stride))
        layers = lens.source_layers[::stride]
        if not layers:
            raise ValueError("No layers selected for the requested stride.")

        compare_top_k = max(1, int(compare_top_k))
        marker_top_k = max(1, int(marker_top_k))

        cols = list(range(len(positions)))
        rows = list(range(len(layers)))

        source_tokens_full = [decode_token(token_ids[p]) for p in positions]
        target_tokens_full = [decode_token(token_ids[p + 1]) for p in positions]
        source_ticktext = [truncate_text(t, axis_token_limit) for t in source_tokens_full]
        target_ticktext = [truncate_text(t, axis_token_limit) for t in target_tokens_full]
        target_token_ids = {p: token_ids[p + 1] for p in positions}

        jl_logits, final_logits, _ = lens.apply(
            model,
            prompt_text,
            layers=layers,
            positions=positions,
            use_jacobian=True,
        )
        logit_logits, _, _ = lens.apply(
            model,
            prompt_text,
            layers=layers,
            positions=positions,
            use_jacobian=False,
        )

        lens_logits_by_name: dict[str, dict[int, torch.Tensor]] = {
            "jlens": jl_logits,
            "logit": logit_logits,
        }
        if tuned_lens is not None:
            lens_logits_by_name["tuned"] = compute_tuned_logits(
                input_ids_full,
                positions,
                layers,
            )

        return {
            "layers": layers,
            "rows": rows,
            "cols": cols,
            "positions": positions,
            "source_tokens_full": source_tokens_full,
            "target_tokens_full": target_tokens_full,
            "source_ticktext": source_ticktext,
            "target_ticktext": target_ticktext,
            "target_token_ids": target_token_ids,
            "final_logits": final_logits,
            "lens_logits_by_name": lens_logits_by_name,
            "compare_top_k": compare_top_k,
            "marker_top_k": marker_top_k,
        }

    def build_metric_payload(
        state: dict[str, object],
        metric: str,
        left_key: str,
        right_key: str,
    ) -> dict[str, object]:
        layers = state["layers"]
        positions = state["positions"]
        left_logits = state["lens_logits_by_name"][left_key]
        right_logits = state["lens_logits_by_name"][right_key]
        final_logits = state["final_logits"]
        compare_top_k = state["compare_top_k"]
        marker_top_k = state["marker_top_k"]

        z = np.zeros((len(layers), len(positions)), dtype=float)
        text = np.empty((len(layers), len(positions)), dtype=object)
        hover = np.empty((len(layers), len(positions)), dtype=object)
        marker_cases = np.empty((len(layers), len(positions)), dtype=object)
        marker_cases[:] = None

        for row, layer in enumerate(layers):
            for col, pos in enumerate(positions):
                left = left_logits[layer][col]
                right = right_logits[layer][col]
                final = final_logits[col]

                left_ids = topk_ids(left, compare_top_k)
                right_ids = topk_ids(right, compare_top_k)
                final_ids = topk_ids(final, compare_top_k)

                left_toks = [decode_token(t) for t in left_ids]
                right_toks = [decode_token(t) for t in right_ids]
                final_toks = [decode_token(t) for t in final_ids]
                gt_id = state["target_token_ids"][pos]

                val = metric_value(metric, left, right, final, compare_top_k)
                z[row, col] = val
                text[row, col] = (
                    f"{truncate_text(left_toks[0], cell_token_limit)} | "
                    f"{truncate_text(right_toks[0], cell_token_limit)}"
                )
                hover[row, col] = (
                    f"Layer: L{layer}<br>"
                    f"Source index: {pos}<br>"
                    f"Target index: {pos + 1}<br>"
                    f"Source token: {state['source_tokens_full'][col]!r}<br>"
                    f"Target token: {decode_token(gt_id)!r}<br>"
                    f"Metric: {metric_label(metric, left_key, right_key)}<br>"
                    f"Value: {val:.6f}<br><br>"
                    f"{lens_names[left_key]} top-{compare_top_k}: {left_toks}<br>"
                    f"{lens_names[right_key]} top-{compare_top_k}: {right_toks}<br>"
                    f"Final top-{compare_top_k}: {final_toks}"
                )

                left_top1 = left_ids[0] == gt_id
                right_top1 = right_ids[0] == gt_id
                left_topk = gt_id in left_ids[:marker_top_k]
                right_topk = gt_id in right_ids[:marker_top_k]

                if left_top1 and right_top1:
                    marker_cases[row, col] = "both_top1"
                elif left_top1:
                    marker_cases[row, col] = "left_top1"
                elif right_top1:
                    marker_cases[row, col] = "right_top1"
                elif left_topk and right_topk:
                    marker_cases[row, col] = "both_topk"
                elif left_topk:
                    marker_cases[row, col] = "left_topk"
                elif right_topk:
                    marker_cases[row, col] = "right_topk"

        style = metric_style(metric, z)
        return {
            "z": z,
            "text": text,
            "hover": hover,
            "marker_cases": marker_cases,
            "colorscale": style["colorscale"],
            "zmin": style["zmin"],
            "zmax": style["zmax"],
            "label": metric_label(metric, left_key, right_key),
        }

    def build_marker_shapes(
        marker_cases: np.ndarray,
        rows: list[int],
        cols: list[int],
        left_color: str,
        right_color: str,
        both_color: str,
    ) -> list[dict[str, object]]:
        shapes = []
        for row, y in enumerate(rows):
            for col, x in enumerate(cols):
                case = marker_cases[row, col]
                if case is None:
                    continue
                if case == "both_top1":
                    color, dash, width = both_color, "solid", 2.5
                elif case == "left_top1":
                    color, dash, width = left_color, "solid", 2.5
                elif case == "right_top1":
                    color, dash, width = right_color, "solid", 2.5
                elif case == "both_topk":
                    color, dash, width = both_color, "dash", 2.0
                elif case == "left_topk":
                    color, dash, width = left_color, "dash", 2.0
                else:
                    color, dash, width = right_color, "dash", 2.0
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=x - 0.48,
                        x1=x + 0.48,
                        y0=y - 0.48,
                        y1=y + 0.48,
                        line=dict(color=color, width=width, dash=dash),
                        fillcolor="rgba(0,0,0,0)",
                    )
                )
        return shapes

    def build_legend_traces(
        marker_top_k: int,
        left_key: str,
        right_key: str,
        left_color: str,
        right_color: str,
        both_color: str,
    ) -> list[go.Scatter]:
        left_short = lens_short_names[left_key]
        right_short = lens_short_names[right_key]
        return [
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=left_color, width=2.5), name=f"{left_short} top-1"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=right_color, width=2.5), name=f"{right_short} top-1"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=both_color, width=2.5), name="Both top-1"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=left_color, width=2.0, dash="dash"), name=f"{left_short} top-{marker_top_k}"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=right_color, width=2.0, dash="dash"), name=f"{right_short} top-{marker_top_k}"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=both_color, width=2.0, dash="dash"), name=f"Both top-{marker_top_k}"),
        ]

    def current_style_options() -> dict[str, str]:
        return {
            "colorscale": colorscale_dropdown.value,
            "left_color": left_color_dropdown.value,
            "right_color": right_color_dropdown.value,
            "both_color": both_color_dropdown.value,
        }

    def build_figure(
        metric: str,
        state: dict[str, object],
        style_cfg: dict[str, str],
        left_key: str,
        right_key: str,
    ) -> go.Figure:
        payload = build_metric_payload(state, metric, left_key, right_key)
        cols = state["cols"]
        rows = state["rows"]
        layers = state["layers"]
        colorscale = (
            payload["colorscale"]
            if style_cfg["colorscale"] == "Auto"
            else style_cfg["colorscale"]
        )
        shapes = build_marker_shapes(
            payload["marker_cases"],
            rows,
            cols,
            style_cfg["left_color"],
            style_cfg["right_color"],
            style_cfg["both_color"],
        )
        legend_traces = build_legend_traces(
            state["marker_top_k"],
            left_key,
            right_key,
            style_cfg["left_color"],
            style_cfg["right_color"],
            style_cfg["both_color"],
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=payload["z"],
                x=cols,
                y=rows,
                text=payload["text"],
                customdata=payload["hover"],
                hovertemplate="%{customdata}<extra></extra>",
                texttemplate="%{text}",
                textfont={"size": 9},
                xgap=0,
                ygap=0,
                colorscale=colorscale,
                zmin=payload["zmin"],
                zmax=payload["zmax"],
                colorbar=dict(
                    title=dict(text=payload["label"], side="right"),
                    thickness=14,
                    len=0.86,
                    tickfont=dict(size=9),
                    x=1.02,
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cols,
                y=[None] * len(cols),
                xaxis="x2",
                yaxis="y",
                mode="markers",
                marker_opacity=0,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        for tr in legend_traces:
            fig.add_trace(tr)

        fig.update_xaxes(
            type="linear",
            tickmode="array",
            tickvals=cols,
            ticktext=state["source_ticktext"],
            title=dict(text="Token Used To Predict", standoff=6),
            tickfont=dict(size=10),
            side="bottom",
            range=[cols[0] - 0.5, cols[-1] + 0.5],
            showgrid=False,
            zeroline=False,
        )
        fig.update_layout(
            xaxis2=dict(
                overlaying="x",
                anchor="y",
                side="top",
                tickmode="array",
                tickvals=cols,
                ticktext=state["target_ticktext"],
                title=dict(text="Next Token Target", standoff=4),
                tickfont=dict(size=10),
                range=[cols[0] - 0.5, cols[-1] + 0.5],
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                ticks="outside",
            )
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=rows,
            ticktext=[f"L{layer}" for layer in layers],
            tickfont=dict(size=10),
            title=dict(text="Layer Index", standoff=8),
            range=[rows[0] - 0.5, rows[-1] + 0.5],
        )
        fig.update_layout(
            title=None,
            width=figure_width,
            height=figure_height,
            margin=dict(l=70, r=105, t=72, b=104),
            shapes=shapes,
            legend=dict(
                orientation="h",
                x=0.0,
                xanchor="left",
                y=-0.11,
                yanchor="top",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.7)",
                entrywidth=110,
                entrywidthmode="pixels",
                tracegroupgap=0,
            ),
        )
        return fig

    title_block = widgets.HTML(value="")

    prompt_input = widgets.Textarea(
        value=prompt,
        description="Prompt:",
        layout=widgets.Layout(width="100%", height="78px", margin="0 0 4px 0"),
        style={"description_width": "60px"},
    )
    left_options = [(label, key) for key, label in lens_names.items()]
    right_default = "logit" if "logit" in lens_names else next(iter(lens_names))
    metric_dropdown = widgets.Dropdown(
        options=[(label, key) for key, label in metrics.items()],
        value="jaccard_pair",
        description="Metric:",
        layout=compact_layout("210px"),
    )
    left_lens_dropdown = widgets.Dropdown(
        options=left_options,
        value="jlens",
        description="Left:",
        layout=compact_layout("165px"),
    )
    right_lens_dropdown = widgets.Dropdown(
        options=left_options,
        value=right_default,
        description="Right:",
        layout=compact_layout("165px"),
    )
    colorscale_dropdown = widgets.Dropdown(
        options=colorscale_options,
        value="Auto",
        description="Colormap:",
        layout=compact_layout("160px"),
    )
    stride_widget = widgets.BoundedIntText(
        value=max(1, int(layer_stride)),
        min=1,
        max=max(1, len(lens.source_layers)),
        step=1,
        description="Layers:",
        layout=compact_layout("110px"),
    )
    positions_widget = widgets.BoundedIntText(
        value=max(1, int(last_n_positions)),
        min=1,
        max=512,
        step=1,
        description="Last n:",
        layout=compact_layout("120px"),
    )
    start_token_widget = widgets.Text(
        value="",
        placeholder="0",
        description="Start id:",
        layout=compact_layout("130px"),
    )
    end_token_widget = widgets.Text(
        value="",
        placeholder="13",
        description="End id:",
        layout=compact_layout("130px"),
    )
    top_k_widget = widgets.BoundedIntText(
        value=max(1, int(top_k)),
        min=1,
        max=100,
        step=1,
        description="Top-k:",
        layout=compact_layout("110px"),
    )
    correct_k_widget = widgets.BoundedIntText(
        value=max(1, int(correct_k)),
        min=1,
        max=100,
        step=1,
        description="Mark k:",
        layout=compact_layout("110px"),
    )
    left_color_dropdown = widgets.Dropdown(
        options=marker_color_options,
        value="black",
        description="Left:",
        layout=compact_layout("120px"),
    )
    right_color_dropdown = widgets.Dropdown(
        options=marker_color_options,
        value="gray",
        description="Right:",
        layout=compact_layout("120px"),
    )
    both_color_dropdown = widgets.Dropdown(
        options=marker_color_options,
        value="deeppink",
        description="Both:",
        layout=compact_layout("125px"),
    )
    run_button = widgets.Button(
        description="Run",
        button_style="",
        layout=widgets.Layout(width="auto", min_width="110px", margin="0"),
    )
    plot_widget = widgets.HTML(
        value="", layout=widgets.Layout(width="100%", margin="0", padding="0")
    )

    state_cache: dict[str, object] = {"payload": None}

    def pair_keys() -> tuple[str, str]:
        return left_lens_dropdown.value, right_lens_dropdown.value

    def update_title_block() -> None:
        left_key, right_key = pair_keys()
        title_block.value = f"""
        <div style=\"margin:0 0 4px 0;\">
          <div style=\"font-size:20px; font-weight:600;\">{html.escape(lens_names[left_key])} top-1 | {html.escape(lens_names[right_key])} top-1</div>
        </div>
        """

    def render_cached() -> None:
        payload = state_cache["payload"]
        if payload is None:
            return
        left_key, right_key = pair_keys()
        if left_key == right_key:
            plot_widget.value = (
                '<div style="padding:8px 0; color:#900; font-family:monospace;">'
                + html.escape("Choose two different lenses to compare.")
                + "</div>"
            )
            return
        fig = build_figure(
            metric_dropdown.value,
            payload,
            current_style_options(),
            left_key,
            right_key,
        )
        fig_html = fig.to_html(
            full_html=True,
            include_plotlyjs="cdn",
            config={"responsive": False, "displayModeBar": True},
        )
        encoded = base64.b64encode(fig_html.encode("utf-8")).decode("ascii")
        iframe_height = figure_height + 18
        plot_widget.value = f"""
            <div style="width:100%; overflow-x:auto; overflow-y:hidden; margin:0; padding:0;">
              <iframe
                src="data:text/html;base64,{encoded}"
                style="width:{figure_width}px; min-width:{figure_width}px; height:{iframe_height}px; border:0; display:block; margin:0; padding:0; overflow:hidden;"
                loading="eager"
                scrolling="no"
                referrerpolicy="no-referrer"
              ></iframe>
            </div>
        """

    def redraw(*_) -> None:
        try:
            update_title_block()
            left_key, right_key = pair_keys()
            if left_key == right_key:
                raise ValueError("Choose two different lenses to compare.")
            state_cache["payload"] = compute_prompt_state(
                prompt_input.value,
                stride_widget.value,
                positions_widget.value,
                top_k_widget.value,
                correct_k_widget.value,
                start_token_widget.value,
                end_token_widget.value,
            )
            render_cached()
        except Exception as exc:
            plot_widget.value = (
                '<div style="padding:8px 0; color:#900; font-family:monospace;">'
                + html.escape(str(exc))
                + "</div>"
            )

    def rerender_existing(change) -> None:
        del change
        update_title_block()
        if state_cache["payload"] is not None:
            render_cached()

    for widget in [
        metric_dropdown,
        left_lens_dropdown,
        right_lens_dropdown,
        colorscale_dropdown,
        left_color_dropdown,
        right_color_dropdown,
        both_color_dropdown,
    ]:
        widget.observe(rerender_existing, names="value")
    run_button.on_click(redraw)

    grid_layout = lambda cols: widgets.Layout(
        width="100%",
        grid_template_columns=cols,
        grid_gap="4px 8px",
        margin="0",
        align_items="center",
    )
    controls_row_1 = widgets.GridBox(
        [metric_dropdown, left_lens_dropdown, right_lens_dropdown, stride_widget, positions_widget],
        layout=grid_layout("repeat(auto-fit, minmax(140px, 1fr))"),
    )
    controls_row_2 = widgets.GridBox(
        [start_token_widget, end_token_widget, top_k_widget, correct_k_widget],
        layout=grid_layout("repeat(auto-fit, minmax(120px, 1fr))"),
    )
    controls_row_3 = widgets.GridBox(
        [colorscale_dropdown, left_color_dropdown, right_color_dropdown, both_color_dropdown, run_button],
        layout=grid_layout("repeat(auto-fit, minmax(120px, 1fr))"),
    )

    update_title_block()
    return widgets.VBox(
        [title_block, prompt_input, controls_row_1, controls_row_2, controls_row_3, plot_widget],
        layout=widgets.Layout(width="100%", gap="4px"),
    )


def show_lens_comparison_widget(*args, **kwargs):
    return build_lens_comparison_widget(*args, **kwargs)
