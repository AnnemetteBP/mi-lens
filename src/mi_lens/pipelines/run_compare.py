from __future__ import annotations

from pathlib import Path
from typing import Any

from ..methods import compare_group_means, summarize_capture_rows

from .common import load_jsonl_records, write_json


def _load_row_files(row_files: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_file in row_files:
        rows.extend(load_jsonl_records(row_file))
    return rows


def run_compare_single_model_lenses_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    row_files = [Path(path) for path in config["row_files"]]
    rows = _load_row_files(row_files)

    group_by = config.get(
        "group_by",
        ["model_label", "model_variant", "language", "layer", "lens_name"],
    )
    summary_rows = summarize_capture_rows(rows, group_by=group_by)

    compare_pairs = config.get(
        "compare_pairs",
        [["tuned", "logit"], ["jlens", "logit"], ["jlens", "tuned"]],
    )
    metric_fields = config.get(
        "metric_fields",
        [
            "gold_prob",
            "gold_rank",
            "topk_jaccard_vs_final",
            "js_vs_final",
            "tv_vs_final",
            "gt_prob_diff",
            "cosine_to_final",
            "relative_l2_to_final",
        ],
    )
    base_group = [field for field in group_by if field != "lens_name"]
    deltas = []
    for left_value, right_value in compare_pairs:
        deltas.extend(
            compare_group_means(
                summary_rows,
                group_by=base_group,
                compare_key="lens_name",
                left_value=left_value,
                right_value=right_value,
                metric_fields=metric_fields,
            )
        )

    output = {
        "summary_rows": summary_rows,
        "delta_rows": deltas,
        "group_by": group_by,
        "metric_fields": metric_fields,
    }
    if config.get("output_path"):
        write_json(config["output_path"], output)
    return output


def run_compare_model_variants_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    row_files = [Path(path) for path in config["row_files"]]
    rows = _load_row_files(row_files)

    group_by = config.get(
        "group_by",
        ["model_family", "lens_name", "language", "layer", "model_variant"],
    )
    summary_rows = summarize_capture_rows(rows, group_by=group_by)

    metric_fields = config.get(
        "metric_fields",
        [
            "gold_prob",
            "gold_rank",
            "topk_jaccard_vs_final",
            "js_vs_final",
            "tv_vs_final",
            "gt_prob_diff",
            "cosine_to_final",
            "relative_l2_to_final",
        ],
    )
    base_group = [field for field in group_by if field != "model_variant"]
    deltas = compare_group_means(
        summary_rows,
        group_by=base_group,
        compare_key="model_variant",
        left_value=config.get("left_value", "danish"),
        right_value=config.get("right_value", "base"),
        metric_fields=metric_fields,
    )

    output = {
        "summary_rows": summary_rows,
        "delta_rows": deltas,
        "group_by": group_by,
        "metric_fields": metric_fields,
    }
    if config.get("output_path"):
        write_json(config["output_path"], output)
    return output
