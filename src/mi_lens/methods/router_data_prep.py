"""Fixed, source-order dataset exports for RouterInterp-style analyses.

The exporter deliberately keeps SAE fitting data separate from benchmark rows.
Every emitted row records its original Hugging Face split index, so a later
larger run can extend a slice without replacing examples already analysed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
from datasets import load_dataset

from .data_prep import PreparedDatasetManifest, _sha256_file, _utc_now, _write_jsonl, _write_manifest


# `datasets>=4` no longer executes remote dataset scripts. These are the Hub's
# maintained Parquet conversions of the exact public datasets used by OLMES.
_PARQUET_CONVERSION_CONFIGS = {
    "allenai/social_i_qa": "default",
    "google-research-datasets/mbpp": "full",
}


# Keep the runtime RouterInterp pipeline independent of a local FlexEval clone.
# EuroEval's former ``EuroEval/...`` Hub repos were removed.  These maintained
# public sources provide the same Danish tasks and are validated by the cache
# warm-up stage before any model is loaded.
_EUROEVAL_DANISH_SOURCES = {
    "angry-tweets": "sorenmulli/angry-tweets-mini",
    "scala-da": "alexandrainst/scala",
    "dansk": "sorenmulli/dane-mini",
    "multi-wiki-qa-da": "alexandrainst/multi-wiki-qa",
    "nordjylland-news": "alexandrainst/nordjylland-news-summarization",
    "danske-talemaader": "Juunge/danske-talemaader-QA",
    "danish-citizen-tests": "sorenmulli/citizenship-test-da",
    "hellaswag-da": "alexandrainst/m_hellaswag",
    "ifeval-da": "danish-foundation-models/ifeval-da",
}

@dataclass(slots=True)
class RouterDatasetExportSpec:
    """One fixed dataset slice used for SAE fitting or router evaluation."""

    name: str
    dataset_name: str
    split: str
    role: str
    domain: str
    language: str
    adapter: str
    output_path: str | Path
    config_name: str | None = None
    revision: str | None = None
    data_file: str | None = None
    start_idx: int = 0
    max_records: int | None = 100
    filter_field: str | None = None
    filter_value: Any = None
    text_field: str = "text"
    target_field: str | None = None
    choices_field: str | None = None
    label_field: str | None = "label"
    native_id_field: str | None = None
    prompt_template: str | None = None
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    shuffle_on_load: bool = False
    loader: str = "huggingface"


@dataclass(slots=True)
class RouterDataPaths:
    root: Path
    data_root: Path
    fit_dir: Path
    eval_dir: Path
    registry_path: Path
    cache_dir: Path


def router_data_paths(project_root: str | Path) -> RouterDataPaths:
    root = Path(project_root)
    data_root = root / "data" / "router"
    return RouterDataPaths(
        root=root,
        data_root=data_root,
        fit_dir=data_root / "sae_fit",
        eval_dir=data_root / "eval",
        registry_path=data_root / "registry.json",
        cache_dir=root / "tmp" / "hf_cache",
    )


def _configure_hf_cache(cache_dir: Path) -> None:
    """Keep dataset downloads off the home drive for this project."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    env_path = cache_dir.parents[1] / ".env"
    if not os.environ.get("HF_TOKEN") and env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
                break


def _load_euroeval_dataset(spec: RouterDatasetExportSpec, cache_dir: Path):
    """Load a maintained Danish task source without importing FlexEval."""

    try:
        source = _EUROEVAL_DANISH_SOURCES[spec.dataset_name]
    except KeyError as exc:
        raise KeyError(f"Unknown EuroEval Danish dataset {spec.dataset_name!r}.") from exc
    # Nordjylland has mixed historical schemas in its dataset metadata.  The
    # explicit Parquet file in the config selects the current text/summary data.
    if spec.data_file is not None:
        return load_dataset(
            "parquet",
            data_files={spec.split: spec.data_file},
            split=spec.split,
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
        ), None
    dataset = load_dataset(
        source,
        spec.config_name,
        split=spec.split,
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    return dataset, None


def _load_router_dataset(spec: RouterDatasetExportSpec, cache_dir: Path):
    """Load a Hub dataset, using its Parquet conversion when scripts are retired."""

    if spec.loader == "euroeval":
        return _load_euroeval_dataset(spec, cache_dir)
    if spec.loader != "huggingface":
        raise ValueError(f"Unsupported router dataset loader {spec.loader!r}.")
    if spec.data_file is not None:
        return load_dataset(
            "parquet",
            data_files={spec.split: spec.data_file},
            split=spec.split,
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
        ), None

    conversion_config = _PARQUET_CONVERSION_CONFIGS.get(spec.dataset_name)
    if conversion_config is None:
        return load_dataset(
            spec.dataset_name,
            spec.config_name,
            split=spec.split,
            revision=spec.revision,
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
        ), None

    config_name = spec.config_name or conversion_config
    data_file = (
        f"hf://datasets/{spec.dataset_name}@refs/convert/parquet/"
        f"{config_name}/{spec.split}/*.parquet"
    )
    return load_dataset(
        "parquet",
        data_files={spec.split: data_file},
        split=spec.split,
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN"),
    ), None


def _require_field(item: dict[str, Any], field: str, *, source_idx: int) -> Any:
    if field not in item:
        available = ", ".join(sorted(item))
        raise KeyError(
            f"Expected field {field!r} in row {source_idx}, but found: {available}."
        )
    return item[field]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _format_choices(choices: list[Any]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"{labels[index]}. {_as_text(choice)}" for index, choice in enumerate(choices)
    )


def _normalize_label(label: Any, choices: list[Any] | None = None) -> int | str | None:
    if label is None:
        return None
    if isinstance(label, int):
        return label
    text = _as_text(label).strip()
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return ord(text.upper()) - ord("A")
    if choices is not None:
        for index, choice in enumerate(choices):
            if text == _as_text(choice):
                return index
    return text


def _apply_template(template: str, item: dict[str, Any], *, source_idx: int) -> str:
    class RequiredFields(dict[str, Any]):
        def __missing__(self, key: str) -> Any:
            available = ", ".join(sorted(self))
            raise KeyError(
                f"Prompt template needs field {key!r} in row {source_idx}, "
                f"but found: {available}."
            )

    return template.format_map(RequiredFields({key: _as_text(value) for key, value in item.items()}))


def _build_euroeval_row(
    spec: RouterDatasetExportSpec,
    item: dict[str, Any],
    *,
    source_idx: int,
) -> tuple[str, list[Any] | None, int | str | None, str | None]:
    """Build prompts from the maintained Danish task schemas.

    The source datasets do not share EuroEval's former preprocessed schema, so
    their task-specific fields must be handled explicitly rather than guessed.
    """

    dataset_name = spec.dataset_name
    if dataset_name == "angry-tweets":
        text = _as_text(_require_field(item, "text", source_idx=source_idx))
        return f"Tekst: {text}\nSentiment:", None, _normalize_label(item.get("label")), None
    if dataset_name == "scala-da":
        text = _as_text(_require_field(item, "text", source_idx=source_idx))
        return f"Sætning: {text}\nGrammatisk korrekt:", None, _normalize_label(item.get("label")), None
    if dataset_name == "dansk":
        text = _as_text(_require_field(item, "text", source_idx=source_idx))
        return f"Sætning: {text}\nNavngivne entiteter:", None, None, None
    if dataset_name == "multi-wiki-qa-da":
        context = _as_text(_require_field(item, "context", source_idx=source_idx))
        question = _as_text(_require_field(item, "question", source_idx=source_idx))
        answers = _require_field(item, "answers", source_idx=source_idx)
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        gold_text = _as_text(answer_texts[0]) if answer_texts else None
        return f"Tekst: {context}\nSpørgsmål: {question}\nSvar:", None, None, gold_text
    if dataset_name == "nordjylland-news":
        text = _as_text(_require_field(item, "text", source_idx=source_idx))
        summary = _as_text(_require_field(item, "summary", source_idx=source_idx))
        return f"Dokument: {text}\nResumé:", None, None, summary
    if dataset_name == "danske-talemaader":
        question = _as_text(_require_field(item, "instruction", source_idx=source_idx))
        choices = [
            _require_field(item, "option_a", source_idx=source_idx),
            _require_field(item, "option_b", source_idx=source_idx),
            _require_field(item, "option_c", source_idx=source_idx),
            _require_field(item, "option_d", source_idx=source_idx),
        ]
        return f"Spørgsmål: {question}\n{_format_choices(choices)}\nSvar:", choices, _normalize_label(item.get("answer"), choices), None
    if dataset_name == "danish-citizen-tests":
        question = _as_text(_require_field(item, "question", source_idx=source_idx))
        choices = [
            _require_field(item, "option-A", source_idx=source_idx),
            _require_field(item, "option-B", source_idx=source_idx),
        ]
        option_c = item.get("option-C")
        if option_c is not None:
            choices.append(option_c)
        return f"Spørgsmål: {question}\n{_format_choices(choices)}\nSvar:", choices, _normalize_label(item.get("correct"), choices), None
    if dataset_name == "hellaswag-da":
        context = _as_text(_require_field(item, "ctx", source_idx=source_idx))
        choices = list(_require_field(item, "endings", source_idx=source_idx))
        return f"Fortsæt teksten:\n{context}\n{_format_choices(choices)}\nSvar:", choices, _normalize_label(item.get("label"), choices), None
    if dataset_name == "ifeval-da":
        prompt = _as_text(_require_field(item, "prompt", source_idx=source_idx))
        return f"Opgave: {prompt}\nSvar:", None, None, None
    raise KeyError(f"Unsupported Danish task prompt builder for {dataset_name!r}.")


def _build_row(
    spec: RouterDatasetExportSpec,
    item: dict[str, Any],
    source_idx: int,
    *,
    euroeval_config: Any = None,
) -> dict[str, Any]:
    adapter = spec.adapter
    choices: list[Any] | None = None
    gold: int | str | None = None
    gold_text: str | None = None

    if adapter == "text":
        prompt = _as_text(_require_field(item, spec.text_field, source_idx=source_idx))
    elif adapter == "euroeval":
        prompt, choices, gold, gold_text = _build_euroeval_row(
            spec, item, source_idx=source_idx
        )
    elif adapter == "template":
        if not spec.prompt_template:
            raise ValueError(f"{spec.name}: adapter='template' requires prompt_template.")
        prompt = _apply_template(spec.prompt_template, item, source_idx=source_idx)
    elif adapter == "gsm8k":
        question = _as_text(_require_field(item, "question", source_idx=source_idx))
        prompt = f"Question: {question}\nAnswer:"
        gold_text = _as_text(_require_field(item, "answer", source_idx=source_idx))
    elif adapter == "hellaswag":
        ctx_a = _as_text(_require_field(item, "ctx_a", source_idx=source_idx))
        ctx_b = _as_text(_require_field(item, "ctx_b", source_idx=source_idx)).capitalize()
        activity = _as_text(_require_field(item, "activity_label", source_idx=source_idx))
        choices = list(_require_field(item, "endings", source_idx=source_idx))
        prompt = (
            f"{activity}: {ctx_a} {ctx_b}\nChoose the best continuation:\n"
            f"{_format_choices(choices)}\nAnswer:"
        )
        gold = _normalize_label(item.get("label"), choices)
    elif adapter == "social_iqa":
        context = _as_text(_require_field(item, "context", source_idx=source_idx))
        question = _as_text(_require_field(item, "question", source_idx=source_idx))
        choices = [
            _require_field(item, "answerA", source_idx=source_idx),
            _require_field(item, "answerB", source_idx=source_idx),
            _require_field(item, "answerC", source_idx=source_idx),
        ]
        prompt = f"Question: {context} {question}\n{_format_choices(choices)}\nAnswer:"
        label = _normalize_label(item.get("label"), choices)
        gold = label - 1 if isinstance(label, int) else label
    elif adapter == "mmlu_pro":
        question = _as_text(_require_field(item, "question", source_idx=source_idx))
        choices = list(_require_field(item, "options", source_idx=source_idx))
        prompt = f"Question: {question}\n{_format_choices(choices)}\nAnswer:"
        gold = _normalize_label(item.get("answer_index"), choices)
    elif adapter == "mbpp":
        prompt = _as_text(_require_field(item, "text", source_idx=source_idx))
        gold_text = _as_text(item.get("code")) or None
    elif adapter == "muse_news":
        article = re.sub(r"\s+", " ", _as_text(_require_field(item, "text", source_idx=source_idx))).strip()
        words = article.split()
        prompt = "Breaking News:\n" + " ".join(words[:32])
        gold_text = " ".join(words[32:]) or None
    elif adapter == "poetry":
        poem = _as_text(_require_field(item, "content", source_idx=source_idx)).strip()
        words = poem.split()
        prompt = "Continue the poem:\n" + " ".join(words[:32])
        gold_text = " ".join(words[32:]) or None
    elif adapter == "multiple_choice":
        text = _as_text(_require_field(item, spec.text_field, source_idx=source_idx))
        if not spec.choices_field:
            raise ValueError(f"{spec.name}: adapter='multiple_choice' requires choices_field.")
        raw_choices = _require_field(item, spec.choices_field, source_idx=source_idx)
        if not isinstance(raw_choices, list):
            raise TypeError(f"{spec.name}: {spec.choices_field!r} must contain a list.")
        choices = raw_choices
        prompt = f"{text}\n{_format_choices(choices)}\nAnswer:"
        if spec.label_field:
            gold = _normalize_label(item.get(spec.label_field), choices)
    else:
        raise ValueError(f"Unsupported router dataset adapter {adapter!r}.")

    if adapter in {"text", "template"}:
        if spec.target_field:
            gold_text = _as_text(item.get(spec.target_field)) or None
        if spec.label_field:
            gold = _normalize_label(item.get(spec.label_field))

    native_id = item.get(spec.native_id_field) if spec.native_id_field else None
    return {
        "id": f"{spec.name}:{source_idx}",
        # Keep a simple, stable label alongside the full source provenance so
        # pooled SAE captures can later be partitioned by dataset.
        "dataset_name": spec.name,
        "prompt": f"{spec.prompt_prefix}{prompt.strip()}{spec.prompt_suffix}",
        "task": spec.name,
        "domain": spec.domain,
        "language": spec.language,
        "role": spec.role,
        "adapter": adapter,
        "choices": choices,
        "gold_choice": gold,
        "gold_text": gold_text,
        "source": {
            "dataset": spec.dataset_name,
            "config": spec.config_name,
            "revision": spec.revision,
            "data_file": spec.data_file,
            "split": spec.split,
            "index": source_idx,
            "native_id": native_id,
        },
    }


def _matches_spec(spec: RouterDatasetExportSpec, item: dict[str, Any]) -> bool:
    if spec.filter_field is not None and item.get(spec.filter_field) != spec.filter_value:
        return False
    if spec.adapter == "muse_news":
        word_count = len(_as_text(item.get("text")).split())
        return 64 <= word_count <= 128
    if spec.adapter == "poetry":
        poem = _as_text(item.get("content"))
        return 64 <= len(poem.split()) <= 128 and "\r\n\r\n" not in poem
    return True


def export_router_dataset_to_jsonl(
    spec: RouterDatasetExportSpec,
    *,
    cache_dir: str | Path,
) -> PreparedDatasetManifest:
    """Export one fixed source-order dataset slice for router analysis."""

    if spec.shuffle_on_load:
        raise ValueError("Router datasets must preserve Hugging Face source order; shuffle is disabled.")
    if spec.role not in {"sae_fit", "eval"}:
        raise ValueError(f"Unsupported router dataset role {spec.role!r}.")
    if spec.max_records is not None and spec.max_records < 1:
        raise ValueError("max_records must be at least one.")

    cache_path = Path(cache_dir)
    _configure_hf_cache(cache_path)
    dataset, euroeval_config = _load_router_dataset(spec, cache_path)

    rows: list[dict[str, Any]] = []
    dropped = 0
    start_idx = max(0, int(spec.start_idx))
    last_scanned = start_idx
    for source_idx in range(start_idx, len(dataset)):
        last_scanned = source_idx + 1
        item = dict(dataset[source_idx])
        if not _matches_spec(spec, item):
            continue
        row = _build_row(spec, item, source_idx, euroeval_config=euroeval_config)
        if not row["prompt"]:
            dropped += 1
            continue
        rows.append(row)
        if spec.max_records is not None and len(rows) >= spec.max_records:
            break

    if not rows:
        raise ValueError(
            f"{spec.name}: no rows exported from {spec.dataset_name!r}. "
            "Check split, filter, and field names."
        )

    output_path = Path(spec.output_path)
    _write_jsonl(output_path, rows)
    manifest = PreparedDatasetManifest(
        kind="router_dataset",
        dataset_name=spec.dataset_name,
        config_name=spec.config_name,
        split=spec.split,
        output_path=str(output_path),
        generated_at_utc=_utc_now(),
        datasets_version=datasets.__version__,
        start_idx=start_idx,
        end_idx_exclusive=last_scanned,
        records_written=len(rows),
        dropped_records=dropped,
        sha256=_sha256_file(output_path),
        extra={
            "router_spec": asdict(spec) | {"output_path": str(output_path)},
            "source_order": True,
            "filter_applied": spec.filter_field is not None or spec.adapter in {"muse_news", "poetry"},
        },
    )
    _write_manifest(output_path, manifest)
    return manifest


def prepare_router_datasets_from_config(
    config_path: str | Path,
    *,
    project_root: str | Path,
    roles: set[str] | None = None,
    continue_on_error: bool = False,
) -> list[PreparedDatasetManifest]:
    """Materialize enabled SAE-fit/evaluation specs from a JSON configuration."""

    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    paths = router_data_paths(project_root)
    selected_roles = roles or {"sae_fit", "eval"}
    manifests: list[PreparedDatasetManifest] = []
    failures: list[dict[str, str]] = []
    previous_datasets: list[dict[str, Any]] = []
    previous_failures: list[dict[str, Any]] = []
    if paths.registry_path.exists():
        previous_registry = json.loads(paths.registry_path.read_text(encoding="utf-8"))
        previous_datasets = list(previous_registry.get("datasets", []))
        previous_failures = list(previous_registry.get("failures", []))
    for role, destination in (("sae_fit", paths.fit_dir), ("eval", paths.eval_dir)):
        if role not in selected_roles:
            continue
        for raw_spec in config.get(role, []):
            if not raw_spec.get("enabled", True):
                continue
            values = dict(raw_spec)
            values.pop("enabled", None)
            values["role"] = role
            values["output_path"] = destination / f"{values['name']}.jsonl"
            spec = RouterDatasetExportSpec(**values)
            try:
                manifests.append(export_router_dataset_to_jsonl(spec, cache_dir=paths.cache_dir))
            except Exception as error:
                if not continue_on_error:
                    raise
                failures.append(
                    {
                        "name": spec.name,
                        "role": role,
                        "dataset": spec.dataset_name,
                        "split": spec.split,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )

    registry_path = paths.registry_path
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    preserved_datasets = [
        item
        for item in previous_datasets
        if item.get("extra", {}).get("router_spec", {}).get("role") not in selected_roles
    ]
    preserved_failures = [
        item
        for item in previous_failures
        if item.get("role") is not None and item.get("role") not in selected_roles
    ]
    registry_path.write_text(
        json.dumps(
            {
                "generated_at_utc": _utc_now(),
                "source_order": True,
                "config_path": str(config_path),
                "datasets": preserved_datasets + [manifest.to_dict() for manifest in manifests],
                "failures": preserved_failures + failures,
                "deferred": config.get("deferred", []),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifests


def iter_router_records_from_config(config_path: str | Path, *, project_root: str | Path, role: str):
    """Stream source-order router rows from providers without creating JSONL files."""

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    paths = router_data_paths(project_root)
    _configure_hf_cache(paths.cache_dir)
    for raw_spec in config.get(role, []):
        if not raw_spec.get("enabled", True):
            continue
        values = dict(raw_spec)
        values.pop("enabled", None)
        values["role"] = role
        values["output_path"] = paths.data_root / ".streaming_unused.jsonl"
        spec = RouterDatasetExportSpec(**values)
        dataset, euroeval_config = _load_router_dataset(spec, paths.cache_dir)
        emitted = 0
        for source_idx in range(max(0, int(spec.start_idx)), len(dataset)):
            item = dict(dataset[source_idx])
            if not _matches_spec(spec, item):
                continue
            row = _build_row(spec, item, source_idx, euroeval_config=euroeval_config)
            if row["prompt"]:
                yield row
                emitted += 1
            if spec.max_records is not None and emitted >= spec.max_records:
                break


def warm_router_dataset_cache_from_config(
    config_path: str | Path,
    *,
    project_root: str | Path,
    roles: tuple[str, ...] = ("sae_fit", "eval"),
) -> dict[str, Any]:
    """Download/open all configured source datasets before loading any model.

    Dataset Arrow/Parquet files stay in ``tmp/hf_cache``. This records only a
    compact manifest and deliberately does not export prompt JSONL files.
    """

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = router_data_paths(project_root)
    _configure_hf_cache(paths.cache_dir)
    datasets_info: list[dict[str, Any]] = []
    for role in roles:
        for raw_spec in config.get(role, []):
            if not raw_spec.get("enabled", True):
                continue
            values = dict(raw_spec)
            values.pop("enabled", None)
            values["role"] = role
            values["output_path"] = paths.data_root / ".streaming_unused.jsonl"
            spec = RouterDatasetExportSpec(**values)
            dataset, euroeval_config = _load_router_dataset(spec, paths.cache_dir)
            first_row: dict[str, Any] | None = None
            for source_idx in range(max(0, int(spec.start_idx)), len(dataset)):
                item = dict(dataset[source_idx])
                if _matches_spec(spec, item):
                    first_row = _build_row(
                        spec,
                        item,
                        source_idx,
                        euroeval_config=euroeval_config,
                    )
                    break
            if first_row is None or not first_row["prompt"]:
                raise ValueError(
                    f"{spec.name}: source loaded but no valid prompt could be built "
                    f"from {spec.dataset_name!r} split {spec.split!r}."
                )
            datasets_info.append(
                {
                    "role": role,
                    "name": spec.name,
                    "dataset": spec.dataset_name,
                    "split": spec.split,
                    "source_records": len(dataset),
                    "validated_prompt_characters": len(str(first_row["prompt"])),
                }
            )
    return {
        "dataset_config_path": str(config_path),
        "cache_dir": str(paths.cache_dir),
        "datasets": datasets_info,
    }
