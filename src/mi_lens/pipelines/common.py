from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mi_lens.model_paths import resolve_flexmore_checkpoint


@dataclass(slots=True)
class ModelLoadConfig:
    model_name: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    flexmore_checkpoint: str | None = None
    flexmore_model_root: str | None = None
    tokenizer_name: str | None = None
    tokenizer_type: str | None = None
    use_fast_tokenizer: bool | None = None
    fix_mistral_regex: bool = True


@dataclass(slots=True)
class CaptureOutputPaths:
    family_dir: Path
    compatibility_dir: Path
    jlens_dir: Path
    tuned_lens_dir: Path
    capture_dir: Path
    capture_rows_dir: Path
    capture_tensors_dir: Path
    summaries_dir: Path
    metadata_dir: Path


@dataclass(slots=True)
class PipelineProjectPaths:
    root: Path
    cache_dir: Path
    outputs_dir: Path
    figures_dir: Path

    def for_family(self, family: str) -> CaptureOutputPaths:
        family_dir = self.outputs_dir / family
        return CaptureOutputPaths(
            family_dir=family_dir,
            compatibility_dir=family_dir / "compatibility",
            jlens_dir=family_dir / "jlens",
            tuned_lens_dir=family_dir / "tuned_lens",
            capture_dir=family_dir / "captures",
            capture_rows_dir=family_dir / "captures" / "rows",
            capture_tensors_dir=family_dir / "captures" / "tensors",
            summaries_dir=family_dir / "summaries",
            metadata_dir=family_dir / "metadata",
        )


def pipeline_project_paths(project_root: str | Path) -> PipelineProjectPaths:
    root = Path(project_root)
    return PipelineProjectPaths(
        root=root,
        cache_dir=root / "tmp" / "hf_cache",
        outputs_dir=root / "outputs",
        figures_dir=root / "figures",
    )


def slugify(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out).strip("_") or "item"


def configure_hf_cache(project_root: str | Path) -> Path:
    paths = pipeline_project_paths(project_root)
    cache_dir = paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = paths.root / "tmp" / "runtime"
    xdg_cache_dir = paths.root / "tmp" / "xdg_cache"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints can live on a read-only shared UCloud mount. Keep every cache,
    # lock, and temporary file created by this process inside the project clone.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(cache_dir / "assets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)
    os.environ["TMPDIR"] = str(runtime_dir)
    os.environ["TMP"] = str(runtime_dir)
    os.environ["TEMP"] = str(runtime_dir)
    return cache_dir


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {dtype_name!r}")


def load_model_and_tokenizer(
    config: ModelLoadConfig,
    *,
    project_root: str | Path,
):
    cache_dir = configure_hf_cache(project_root)
    import transformers

    model_source = config.model_name
    if config.flexmore_checkpoint:
        model_source = str(
            resolve_flexmore_checkpoint(
                config.flexmore_checkpoint,
                model_root=config.flexmore_model_root,
            )
        )
        if not Path(model_source).is_dir():
            raise FileNotFoundError(
                f"FlexMoRE checkpoint directory not found: {model_source}. "
                "Set MI_LENS_FLEXMORE_MODEL_ROOT when this UCloud path is mounted elsewhere."
            )

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=resolve_torch_dtype(config.dtype),
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=config.low_cpu_mem_usage,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "trust_remote_code": config.trust_remote_code,
    }
    if config.tokenizer_type is not None:
        tokenizer_kwargs["tokenizer_type"] = config.tokenizer_type
    if config.use_fast_tokenizer is not None:
        tokenizer_kwargs["use_fast"] = config.use_fast_tokenizer

    tokenizer_source = config.tokenizer_name or model_source
    if config.fix_mistral_regex:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_source,
                fix_mistral_regex=True,
                **tokenizer_kwargs,
            )
        except TypeError:
            # Older or forked Transformers versions may not expose this option.
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_source,
                **tokenizer_kwargs,
            )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_source,
            **tokenizer_kwargs,
        )
    return hf_model, tokenizer, cache_dir


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def prompt_text_from_record(record: dict[str, Any]) -> str:
    if "text" in record:
        return str(record["text"])
    if "prompt" in record:
        return str(record["prompt"])
    raise KeyError("Each JSONL row must contain either 'text' or 'prompt'.")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def clear_runtime_state(*objects_to_delete: Any) -> None:
    for obj in objects_to_delete:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def metadata_payload(
    *,
    model_config: ModelLoadConfig,
    cache_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import transformers

    payload = {
        "model": asdict(model_config),
        "cache_dir": str(cache_dir),
        "torch_version": torch.__version__,
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "transformers_module_path": str(Path(transformers.__file__).resolve()),
    }
    if extra:
        payload.update(extra)
    return payload
