from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from jlens import from_hf
from tuned_lens.model_surgery import (
    detect_architecture,
    find_final_norm,
    get_unembedding_matrix,
)

from .batch_capture import BatchCaptureConfig, capture_lens_batch


@dataclass(slots=True)
class CompatibilityCheck:
    name: str
    ok: bool
    detail: str
    rows_captured: int = 0


@dataclass(slots=True)
class CompatibilityReport:
    model_label: str
    model_type: str
    architecture: str
    n_layers: int
    vocab_size: int
    final_norm_found: bool
    checks: list[CompatibilityCheck]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["all_ok"] = all(check.ok for check in self.checks)
        return payload


def _take_smoke_prompts(tokenizer, prompts, max_prompts: int) -> list[str]:
    selected: list[str] = []
    for prompt in prompts:
        if len(selected) >= max_prompts:
            break
        if len(tokenizer(prompt).input_ids) >= 2:
            selected.append(prompt)
    if not selected:
        raise ValueError("Need at least one prompt that tokenizes to 2+ tokens.")
    return selected


def build_compatibility_report(
    hf_model,
    tokenizer,
    prompts,
    *,
    model_label: str,
    jlens_lens=None,
    tuned_lens=None,
    max_prompts: int = 3,
    layer_stride: int = 0,
    last_n_positions: int = 2,
    top_k: int = 5,
) -> CompatibilityReport:
    smoke_prompts = _take_smoke_prompts(tokenizer, prompts, max_prompts=max_prompts)
    architecture = detect_architecture(hf_model)
    final_norm_found = find_final_norm(hf_model) is not None
    unembed = get_unembedding_matrix(hf_model)
    lens_model = from_hf(hf_model, tokenizer)

    n_layers = int(lens_model.n_layers)
    stride = layer_stride if layer_stride > 0 else max(1, n_layers // 4)
    cfg = BatchCaptureConfig(
        layer_stride=stride,
        last_n_positions=last_n_positions,
        top_k=top_k,
        include_topk_tokens=False,
        include_surface_metrics=False,
    )

    checks: list[CompatibilityCheck] = []

    try:
        capture_rows = capture_lens_batch(
            hf_model,
            tokenizer,
            smoke_prompts,
            model_label=model_label,
            lens_model=lens_model,
            lens_names=("logit",),
            config=cfg,
        )
        checks.append(
            CompatibilityCheck(
                name="logit_lens",
                ok=True,
                detail="Plain logit lens capture succeeded.",
                rows_captured=len(capture_rows),
            )
        )
    except Exception as exc:
        checks.append(
            CompatibilityCheck(
                name="logit_lens",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    if tuned_lens is not None:
        try:
            capture_rows = capture_lens_batch(
                hf_model,
                tokenizer,
                smoke_prompts,
                model_label=model_label,
                lens_model=lens_model,
                tuned_lens=tuned_lens,
                lens_names=("tuned",),
                config=cfg,
            )
            checks.append(
                CompatibilityCheck(
                    name="tuned_lens",
                    ok=True,
                    detail="Tuned-lens capture succeeded.",
                    rows_captured=len(capture_rows),
                )
            )
        except Exception as exc:
            checks.append(
                CompatibilityCheck(
                    name="tuned_lens",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

    if jlens_lens is not None:
        try:
            capture_rows = capture_lens_batch(
                hf_model,
                tokenizer,
                smoke_prompts,
                model_label=model_label,
                lens_model=lens_model,
                jlens_lens=jlens_lens,
                lens_names=("jlens",),
                config=cfg,
            )
            checks.append(
                CompatibilityCheck(
                    name="jlens",
                    ok=True,
                    detail="J-lens capture succeeded.",
                    rows_captured=len(capture_rows),
                )
            )
        except Exception as exc:
            checks.append(
                CompatibilityCheck(
                    name="jlens",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

    return CompatibilityReport(
        model_label=model_label,
        model_type=type(hf_model).__name__,
        architecture=architecture,
        n_layers=n_layers,
        vocab_size=int(unembed.out_features),
        final_norm_found=final_norm_found,
        checks=checks,
    )
