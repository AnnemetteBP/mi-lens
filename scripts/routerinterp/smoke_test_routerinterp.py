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
            {"id": "fit-1", "prompt": "router features test english model analysis data"},
            {"id": "fit-2", "prompt": "dansk router sparse expert one two three"},
            {"id": "fit-3", "prompt": "model analysis data router features test"},
            {"id": "fit-4", "prompt": "expert one sparse dansk router features"},
        ],
    )
    _write_jsonl(
        eval_path,
        [
            {"id": "eval-1", "prompt": "router test model data english analysis"},
            {"id": "eval-2", "prompt": "dansk expert sparse features one two"},
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
        "max_tokens": 12,
    }
    analysis_config = {
        "project_root": str(ROOT),
        "train_artifacts_path": "tmp/routerinterp/smoke_test/probe_fit",
        "eval_artifacts_path": "tmp/routerinterp/smoke_test/eval",
        "pretrained_sae_dir": "tmp/routerinterp/smoke_test/streamed_sae",
        "output_path": "tmp/routerinterp/smoke_test/analysis",
        "layers": "captured",
        "expert_labels": ["public", "code", "math", "danish"],
        "protocol": "custom_smoke",
        "max_sae_fit_tokens": 16,
        "max_probe_fit_tokens": 16,
        "max_eval_tokens": 12,
        "active_feature_counts": [1, 2],
        "primary_active_features": 2,
        "top_rho_features": 2,
        "sae": {"k": 2, "n_features": 64, "pca_components": 4},
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
