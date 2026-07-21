from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
from datasets import load_dataset


DATASET_NAME_ALIASES = {
    "wikitext": "Salesforce/wikitext",
    "mkqa": "apple/mkqa",
}

DATASET_LOAD_OVERRIDES = {
    "apple/mkqa": {
        "loader_name": "parquet",
        "data_files": {
            "train": "hf://datasets/apple/mkqa@refs/convert/parquet/mkqa/train/*.parquet"
        },
        "split_map": {
            "train": "train",
            "validation": "train",
            "test": "train",
        },
    }
}


@dataclass(slots=True)
class TextExportSpec:
    dataset_name: str
    split: str
    output_path: str | Path
    text_field: str = "text"
    config_name: str | None = None
    start_idx: int = 0
    max_records: int | None = None
    strip_text: bool = True
    drop_empty: bool = True
    shuffle_on_load: bool = False


@dataclass(slots=True)
class MKQAExportSpec:
    output_path: str | Path
    split: str = "validation"
    dataset_name: str = "mkqa"
    config_name: str | None = None
    question_language: str = "da"
    answer_language: str = "da"
    start_idx: int = 0
    max_records: int | None = None
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    shuffle_on_load: bool = False


@dataclass(slots=True)
class PreparedDatasetManifest:
    kind: str
    dataset_name: str
    config_name: str | None
    split: str
    output_path: str
    generated_at_utc: str
    datasets_version: str
    start_idx: int
    end_idx_exclusive: int
    records_written: int
    dropped_records: int
    sha256: str
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProjectDataPaths:
    root: Path
    data_root: Path
    train_fit_dir: Path
    eval_dir: Path
    registry_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(text: str, *, strip_text: bool) -> str:
    if strip_text:
        text = text.strip()
    return text


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, manifest: PreparedDatasetManifest) -> None:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, ensure_ascii=False, indent=2)


def _slice_stop(start_idx: int, max_records: int | None, available: int) -> int:
    if max_records is None:
        return available
    return min(available, start_idx + max_records)


def _resolve_dataset_name(dataset_name: str) -> str:
    if dataset_name.startswith("YOUR_"):
        raise ValueError(
            "Replace placeholder dataset names like 'YOUR_DANISH_TEXT_DATASET' "
            "with a real Hugging Face dataset id or a local dataset path."
        )
    return DATASET_NAME_ALIASES.get(dataset_name, dataset_name)


def _load_dataset_with_overrides(
    dataset_name: str,
    config_name: str | None,
    split: str,
):
    override = DATASET_LOAD_OVERRIDES.get(dataset_name)
    if override is None:
        return load_dataset(dataset_name, config_name, split=split), split

    split_map = override.get("split_map", {})
    loaded_split = split_map.get(split, split)
    dataset = load_dataset(
        override["loader_name"],
        data_files=override["data_files"],
        split=loaded_split,
    )
    return dataset, loaded_split


def project_data_paths(project_root: str | Path) -> ProjectDataPaths:
    """Return the standard mi-lens data layout."""

    root = Path(project_root)
    data_root = root / "data"
    return ProjectDataPaths(
        root=root,
        data_root=data_root,
        train_fit_dir=data_root / "train_fit",
        eval_dir=data_root / "eval",
        registry_path=data_root / "registry.json",
    )


def export_text_split_to_jsonl(spec: TextExportSpec) -> PreparedDatasetManifest:
    """Export a fixed, unshuffled HF text split to JSONL.

    The exported rows preserve source order and store source indices so future
    experiments can extend the sample range without regenerating different
    examples.
    """

    if spec.shuffle_on_load:
        raise ValueError(
            "Load Hugging Face splits in source order for reproducibility. "
            "Shuffle later downstream if needed."
        )

    resolved_dataset_name = _resolve_dataset_name(spec.dataset_name)

    dataset, loaded_split = _load_dataset_with_overrides(
        resolved_dataset_name,
        spec.config_name,
        spec.split,
    )

    start_idx = max(0, int(spec.start_idx))
    stop_idx = _slice_stop(start_idx, spec.max_records, len(dataset))

    rows: list[dict[str, Any]] = []
    dropped = 0
    for source_idx in range(start_idx, stop_idx):
        item = dataset[source_idx]
        text = item.get(spec.text_field)
        if text is None:
            raise KeyError(
                f"Field {spec.text_field!r} not found in dataset row at index {source_idx}."
            )
        if not isinstance(text, str):
            text = str(text)
        text = _normalize_text(text, strip_text=spec.strip_text)
        if spec.drop_empty and not text:
            dropped += 1
            continue
        rows.append(
            {
                "id": f"{spec.dataset_name}:{spec.split}:{source_idx}",
                "text": text,
                "source": {
                    "dataset": resolved_dataset_name,
                    "dataset_requested": spec.dataset_name,
                    "config": spec.config_name,
                    "split": loaded_split,
                    "split_requested": spec.split,
                    "index": source_idx,
                    "text_field": spec.text_field,
                },
            }
        )

    output_path = Path(spec.output_path)
    _write_jsonl(output_path, rows)
    manifest = PreparedDatasetManifest(
        kind="text",
        dataset_name=resolved_dataset_name,
        config_name=spec.config_name,
        split=loaded_split,
        output_path=str(output_path),
        generated_at_utc=_utc_now(),
        datasets_version=datasets.__version__,
        start_idx=start_idx,
        end_idx_exclusive=stop_idx,
        records_written=len(rows),
        dropped_records=dropped,
        sha256=_sha256_file(output_path),
        extra={
            "text_field": spec.text_field,
            "strip_text": spec.strip_text,
            "drop_empty": spec.drop_empty,
            "dataset_requested": spec.dataset_name,
            "split_requested": spec.split,
        },
    )
    _write_manifest(output_path, manifest)
    return manifest


def _first_answer_text(answer_value: Any) -> str | None:
    if answer_value is None:
        return None
    if isinstance(answer_value, str):
        return answer_value
    if isinstance(answer_value, dict):
        if "text" in answer_value and answer_value["text"]:
            return str(answer_value["text"])
        if "aliases" in answer_value and answer_value["aliases"]:
            return str(answer_value["aliases"][0])
        return None
    if isinstance(answer_value, list):
        for item in answer_value:
            text = _first_answer_text(item)
            if text:
                return text
    return None


def export_mkqa_to_jsonl(spec: MKQAExportSpec) -> PreparedDatasetManifest:
    """Export a fixed MKQA slice for held-out bilingual evaluation."""

    if spec.shuffle_on_load:
        raise ValueError(
            "Load Hugging Face splits in source order for reproducibility. "
            "Shuffle later downstream if needed."
        )

    resolved_dataset_name = _resolve_dataset_name(spec.dataset_name)

    dataset, loaded_split = _load_dataset_with_overrides(
        resolved_dataset_name,
        spec.config_name,
        spec.split,
    )

    start_idx = max(0, int(spec.start_idx))
    stop_idx = _slice_stop(start_idx, spec.max_records, len(dataset))

    rows: list[dict[str, Any]] = []
    dropped = 0
    for source_idx in range(start_idx, stop_idx):
        item = dataset[source_idx]
        queries = item.get("queries", {})
        answers = item.get("answers", {})

        question = queries.get(spec.question_language)
        answer = _first_answer_text(answers.get(spec.answer_language))

        if not question or not answer:
            dropped += 1
            continue

        prompt = f"{spec.prompt_prefix}{question}{spec.prompt_suffix}"
        rows.append(
            {
                "id": str(item.get("example_id", f"{spec.split}:{source_idx}")),
                "prompt": prompt,
                "question": question,
                "answer": answer,
                "question_language": spec.question_language,
                "answer_language": spec.answer_language,
                "source": {
                    "dataset": resolved_dataset_name,
                    "dataset_requested": spec.dataset_name,
                    "config": spec.config_name,
                    "split": loaded_split,
                    "split_requested": spec.split,
                    "index": source_idx,
                    "example_id": item.get("example_id"),
                },
            }
        )

    output_path = Path(spec.output_path)
    _write_jsonl(output_path, rows)
    manifest = PreparedDatasetManifest(
        kind="mkqa",
        dataset_name=resolved_dataset_name,
        config_name=spec.config_name,
        split=loaded_split,
        output_path=str(output_path),
        generated_at_utc=_utc_now(),
        datasets_version=datasets.__version__,
        start_idx=start_idx,
        end_idx_exclusive=stop_idx,
        records_written=len(rows),
        dropped_records=dropped,
        sha256=_sha256_file(output_path),
        extra={
            "question_language": spec.question_language,
            "answer_language": spec.answer_language,
            "prompt_prefix": spec.prompt_prefix,
            "prompt_suffix": spec.prompt_suffix,
            "dataset_requested": spec.dataset_name,
            "split_requested": spec.split,
        },
    )
    _write_manifest(output_path, manifest)
    return manifest


def write_dataset_registry(
    output_path: str | Path,
    manifests: list[PreparedDatasetManifest],
    *,
    note: str | None = None,
) -> None:
    """Write a top-level registry for all prepared datasets."""

    output_path = Path(output_path)
    _ensure_parent(output_path)
    payload = {
        "generated_at_utc": _utc_now(),
        "note": note,
        "datasets": [manifest.to_dict() for manifest in manifests],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
