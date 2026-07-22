#!/usr/bin/env python3
"""Run the complete RouterInterp flow against a tiny local OLMoE model.

This is an offline smoke test: it creates a randomly initialized four-layer
OLMoE checkpoint and a minimal tokenizer under ``tmp/``.  It validates the
same model loader, router hooks, streamed SAE fitting, retained captures, and
analysis code used by the UCloud batch without writing outside the project.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = ROOT / "tmp" / "routerinterp" / "smoke_test"


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_local_model(model_dir: Path) -> None:
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "router": 3,
        "features": 4,
        "test": 5,
        "english": 6,
        "dansk": 7,
        "model": 8,
        "analysis": 9,
        "data": 10,
        "sparse": 11,
        "expert": 12,
        "one": 13,
        "two": 14,
        "three": 15,
    }
    tokenizer_backend = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="[PAD]"))
    tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)

    config = transformers.OlmoeConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    transformers.OlmoeForCausalLM(config).save_pretrained(model_dir)


def _assert_valid_numbers(value, *, path: str = "summary") -> None:
    """Reject silent NaN/Inf and the bounded metrics used by RouterInterp."""

    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_valid_numbers(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_valid_numbers(nested, path=f"{path}[{index}]")
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    if not torch.isfinite(torch.tensor(float(value))):
        raise AssertionError(f"{path} is NaN or infinity")
    bounded = (
        "precision",
        "recall",
        "jaccard",
        "macro_f1",
        "brier",
        "ece",
        "fraction",
        "firing_rate",
        "activation_mass_share",
        "normalized_domain_entropy",
        "total_variation_distance",
    )
    if any(name in path for name in bounded) and not 0.0 <= float(value) <= 1.0:
        raise AssertionError(f"{path} is outside [0, 1]")


def _vendored_matching_pursuit(atoms: torch.Tensor, activations: torch.Tensor, k: int) -> torch.Tensor:
    """Reference the exact vendored ITDA ``mp_encode`` equations."""

    residuals = activations.clone()
    coefficients = torch.zeros(activations.shape[0], atoms.shape[0], dtype=activations.dtype)
    rows = torch.arange(activations.shape[0])
    for _ in range(k):
        correlations = residuals @ atoms.T
        best_atoms = correlations.abs().argmax(dim=1)
        values = correlations[rows, best_atoms]
        coefficients[rows, best_atoms] += values
        residuals -= values.unsqueeze(1) * atoms[best_atoms]
    return coefficients


def _assert_router_distributions(summary: dict[str, object]) -> None:
    """Verify every saved histogram is finite, bounded, and complete."""

    for layer, result in summary["layers"].items():
        distributions = result["router_probability_distribution"]
        token_count = int(distributions["token_count"])
        if token_count <= 0:
            raise AssertionError(f"Layer {layer} has no router-probability tokens.")
        for metric in ("top1_weight", "top1_top2_margin", "normalized_entropy"):
            histogram = distributions[metric]["histogram"]
            edges = [float(value) for value in histogram["edges"]]
            counts = [int(value) for value in histogram["counts"]]
            if edges[0] != 0.0 or edges[-1] != 1.0 or any(
                right <= left for left, right in zip(edges, edges[1:])
            ):
                raise AssertionError(f"Layer {layer} {metric} has invalid histogram edges.")
            if any(count < 0 for count in counts) or sum(counts) != token_count:
                raise AssertionError(f"Layer {layer} {metric} does not account for every token.")


def main() -> None:
    shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
    model_dir = SMOKE_ROOT / "tiny_olmoe"
    model_dir.mkdir(parents=True, exist_ok=True)
    _make_local_model(model_dir)

    fit_path = SMOKE_ROOT / "fit.jsonl"
    eval_path = SMOKE_ROOT / "eval.jsonl"
    _write_jsonl(
        fit_path,
        [
            {"id": "fit-code", "dataset_name": "mbpp", "domain": "code", "language": "en", "prompt": "router features test english model analysis data"},
            {"id": "fit-math", "dataset_name": "gsm8k", "domain": "math", "language": "en", "prompt": "one two three model analysis sparse expert"},
            {"id": "fit-news", "dataset_name": "muse_news", "domain": "news", "language": "en", "prompt": "english model analysis data router features test"},
            {"id": "fit-danish", "dataset_name": "multi_wiki_qa_da", "domain": "knowledge", "language": "da", "prompt": "dansk router expert sparse features one two"},
        ],
    )
    _write_jsonl(
        eval_path,
        [
            {"id": "eval-code", "dataset_name": "mbpp", "domain": "code", "language": "en", "prompt": "router features test model data"},
            {"id": "eval-math", "dataset_name": "gsm8k", "domain": "math", "language": "en", "prompt": "one two three model analysis"},
            {"id": "eval-news", "dataset_name": "muse_news", "domain": "news", "language": "en", "prompt": "english model data analysis features"},
            {"id": "eval-social", "dataset_name": "angry_tweets_da", "domain": "social", "language": "da", "prompt": "dansk router expert features test"},
            {"id": "eval-creative", "dataset_name": "poetry", "domain": "creative", "language": "en", "prompt": "english sparse features model test"},
            {"id": "eval-knowledge", "dataset_name": "multi_wiki_qa_da", "domain": "knowledge", "language": "da", "prompt": "dansk model data router analysis"},
        ],
    )

    model = {
        "model_name": str(model_dir),
        "dtype": "float32",
        "device": "cpu",
        "low_cpu_mem_usage": True,
    }
    shared = {
        "project_root": str(ROOT),
        "model": model,
        "layers": [0, 1, 2, 3],
        "expert_labels": ["public", "code", "math", "danish"],
        "max_seq_len": 32,
        "skip_first_token": False,
        "artifact_dtype": "float32",
    }
    configs_dir = SMOKE_ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    fit_config = {
        **shared,
        "examples_glob": str(fit_path.relative_to(ROOT)),
        "output_path": "tmp/routerinterp/smoke_test/streamed_sae",
        "protocol": "custom_smoke",
        "max_sae_fit_tokens": 16,
        "sae": {"k": 2, "n_features": 64, "batch_size": 4, "learning_rate": 1e-3},
        "itda": {"enabled": True, "fit_tokens": 8, "batch_size": 8, "max_atoms": 32, "k": 2, "loss_threshold": 0.1},
    }
    probe_config = {
        **shared,
        "examples_path": str(fit_path.relative_to(ROOT)),
        "output_path": "tmp/routerinterp/smoke_test/probe_fit",
        "max_tokens": 16,
    }
    eval_config = {
        **shared,
        "examples_path": str(eval_path.relative_to(ROOT)),
        "output_path": "tmp/routerinterp/smoke_test/eval",
        "max_tokens": 48,
    }
    analysis_config = {
        "project_root": str(ROOT),
        "model_label": "Tiny OLMoE router -- layout validation only",
        "train_artifacts_path": "tmp/routerinterp/smoke_test/probe_fit",
        "eval_artifacts_path": "tmp/routerinterp/smoke_test/eval",
        "pretrained_sae_dir": "tmp/routerinterp/smoke_test/streamed_sae",
        "output_path": "tmp/routerinterp/smoke_test/analysis",
        "layers": "captured",
        "expert_labels": ["public", "code", "math", "danish"],
        "protocol": "custom_smoke",
        "max_sae_fit_tokens": 16,
        "max_probe_fit_tokens": 16,
        "max_eval_tokens": 48,
        "active_feature_counts": [1, 2],
        "primary_active_features": 2,
        "top_rho_features": 2,
        "sae": {"k": 2, "n_features": 64, "pca_components": 4},
        "itda": {"enabled": True, "fit_tokens": 8, "batch_size": 8, "max_atoms": 32, "k": 2, "loss_threshold": 0.1},
        "probe": {"steps": 2, "learning_rate": 1e-3},
        "device": "cpu",
        "seed": 0,
    }
    paths = {
        "fit_sae": configs_dir / "fit_sae.json",
        "capture_probe": configs_dir / "capture_probe.json",
        "capture_eval": configs_dir / "capture_eval.json",
        "analysis": configs_dir / "analysis.json",
    }
    for path, payload in zip(
        paths.values(), (fit_config, probe_config, eval_config, analysis_config), strict=True
    ):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    batch_path = configs_dir / "batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "project_root": str(ROOT),
                "batch_name": "routerinterp_smoke_test",
                "continue_on_error": False,
                "jobs": [{"name": "tiny_olmoe", **{f"{stage}_config": str(path) for stage, path in paths.items()}}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "routerinterp" / "run_routerinterp_batch.py"), "--config", str(batch_path)],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    fit_result = json.loads((SMOKE_ROOT / "streamed_sae" / "sae_fit_manifest.json").read_text(encoding="utf-8"))
    probe_result = json.loads((SMOKE_ROOT / "probe_fit" / "manifest.json").read_text(encoding="utf-8"))
    eval_result = json.loads((SMOKE_ROOT / "eval" / "manifest.json").read_text(encoding="utf-8"))
    analysis_result = json.loads((SMOKE_ROOT / "analysis" / "summary.json").read_text(encoding="utf-8"))
    if not fit_result["itda"]["enabled"]:
        raise AssertionError("The smoke test did not fit ITDA.")
    for layer in analysis_result["layers"]:
        layer_dir = SMOKE_ROOT / "analysis" / f"layer_{int(layer):02d}"
        if not (SMOKE_ROOT / "streamed_sae" / f"layer_{int(layer):02d}" / "itda.pt").is_file():
            raise AssertionError(f"Missing ITDA checkpoint for layer {layer}.")
        if not (layer_dir / "itda_analysis.json").is_file():
            raise AssertionError(f"Missing ITDA analysis for layer {layer}.")
        if not (layer_dir / "topk_sae_top_rho_contexts.json").is_file():
            raise AssertionError(f"Missing Top-K SAE contexts for layer {layer}.")
        routing = analysis_result["layers"][layer]["routing_prediction"]
        if routing["itda_predictor"] is None:
            raise AssertionError(f"Missing ITDA routing predictor for layer {layer}.")
        for key in ("domain_expert_activation", "sae_predicted_domain_expert_activation"):
            grouped = analysis_result["layers"][layer][key]
            if not grouped["groups"]:
                raise AssertionError(f"Missing {key} groups for layer {layer}.")
            for group in grouped["groups"]:
                rates = [float(value) for value in group["expert_activation_rate"]]
                if any(rate < 0.0 or rate > 1.0 for rate in rates):
                    raise AssertionError(f"{key} contains an out-of-range activation rate.")
    _assert_valid_numbers(analysis_result)
    _assert_router_distributions(analysis_result)
    itda_checkpoint = SMOKE_ROOT / "streamed_sae" / "layer_00" / "itda.pt"
    from mi_lens.sparse import ITDA

    itda = ITDA.from_pretrained(itda_checkpoint)
    reference_input = torch.tensor([[1.0] + [0.0] * 31, [0.0, 1.0] + [0.0] * 30])
    actual = itda.encode(reference_input)
    expected = _vendored_matching_pursuit(itda.atoms.cpu(), reference_input, itda.config.k)
    if not torch.allclose(actual.cpu(), expected, atol=1e-6, rtol=1e-6):
        raise AssertionError("mi_lens ITDA matching pursuit diverges from the vendored ITDA equations.")
    summary_path = SMOKE_ROOT / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(
            {"fit": fit_result, "probe": probe_result, "eval": eval_result, "analysis": analysis_result},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
