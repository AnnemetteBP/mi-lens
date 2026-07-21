from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..methods.tokenization_audit import (
    TokenAuditConfig,
    build_token_audit_rows,
    pair_token_audit_rows,
    summarize_paired_token_audit_rows,
    summarize_token_audit_rows,
)
from .common import (
    ModelLoadConfig,
    configure_hf_cache,
    load_jsonl_records,
    metadata_payload,
    pipeline_project_paths,
    slugify,
    write_json,
)


def _load_tokenizer(model_cfg: ModelLoadConfig, *, project_root: str | Path):
    cache_dir = configure_hf_cache(project_root)
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=model_cfg.trust_remote_code,
    )
    return tokenizer, cache_dir


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_tokenization_audit_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    model_cfg = ModelLoadConfig(**config["model"])
    model_family = config["model_family"]
    model_label = config.get("model_label") or slugify(model_cfg.model_name)
    model_variant = config.get("model_variant")
    dataset_name = config.get("dataset_name")

    paths = pipeline_project_paths(project_root).for_family(model_family)
    audit_dir = paths.metadata_dir / "tokenization_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, cache_dir = _load_tokenizer(model_cfg, project_root=project_root)
    cfg = TokenAuditConfig(
        budgets=tuple(config.get("budgets", [128, 256, 512])),
        text_keys=tuple(config.get("text_keys", ["prompt", "text", "question"])),
        add_special_tokens=bool(config.get("add_special_tokens", True)),
    )

    examples_by_language = config["examples_by_language"]
    row_files: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_language: dict[str, list[dict[str, Any]]] = {}

    for language, rel_path in examples_by_language.items():
        records = load_jsonl_records(project_root / rel_path)
        max_examples = config.get("max_examples")
        if max_examples is not None:
            records = records[: int(max_examples)]

        rows = build_token_audit_rows(
            tokenizer,
            records,
            model_label=model_label,
            language=language,
            config=cfg,
        )
        rows_by_language[language] = rows
        summaries[language] = summarize_token_audit_rows(rows)
        row_path = audit_dir / f"{model_label}_{language}_token_audit.jsonl"
        _write_jsonl(row_path, rows)
        row_files[language] = str(row_path)

    pair_summaries: dict[str, dict[str, Any]] = {}
    pair_files: dict[str, str] = {}
    pair_languages = config.get("pair_languages", [])
    for left_language, right_language in pair_languages:
        left_rows = rows_by_language.get(left_language, [])
        right_rows = rows_by_language.get(right_language, [])
        paired_rows = pair_token_audit_rows(
            left_rows,
            right_rows,
            left_language=left_language,
            right_language=right_language,
        )
        pair_key = f"{left_language}_vs_{right_language}"
        pair_summaries[pair_key] = summarize_paired_token_audit_rows(paired_rows)
        pair_path = audit_dir / f"{model_label}_{pair_key}_token_pairs.jsonl"
        _write_jsonl(pair_path, paired_rows)
        pair_files[pair_key] = str(pair_path)

    output = {
        "row_files": row_files,
        "pair_files": pair_files,
        "summaries": summaries,
        "pair_summaries": pair_summaries,
        "metadata": metadata_payload(
            model_config=model_cfg,
            cache_dir=cache_dir,
            extra={
                "model_family": model_family,
                "model_label": model_label,
                "model_variant": model_variant,
                "dataset_name": dataset_name,
                "examples_by_language": examples_by_language,
            },
        ),
    }

    summary_path = audit_dir / f"{model_label}_tokenization_audit_summary.json"
    write_json(summary_path, output)
    return {
        "summary_path": str(summary_path),
        "row_files": row_files,
        "pair_files": pair_files,
    }
