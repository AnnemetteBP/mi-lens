from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class SurfaceFormConfig:
    """Configuration for token-surface similarity proxies.

    These metrics are intentionally lightweight. They do not solve semantics,
    but they are more forgiving than exact-token matching when tokenization is
    fragmented or whitespace-prefixed.
    """

    top_k: int = 5
    char_ngram: int = 3
    lowercase: bool = True
    strip_whitespace: bool = True


@dataclass(slots=True)
class LensSurfaceMetrics:
    top1_norm_matches_gold: bool
    top1_norm_matches_final_top1: bool
    topk_contains_norm_gold: bool
    top1_charngram_jaccard_gold: float
    top1_charngram_jaccard_final_top1: float
    best_topk_charngram_jaccard_gold: float
    best_topk_charngram_jaccard_final_top1: float


@dataclass(slots=True)
class ResidualAlignmentMetrics:
    """How well a lens-mapped residual matches the actual final residual."""

    cosine_to_final: float
    l2_to_final: float
    relative_l2_to_final: float
    norm_ratio_to_final: float


@dataclass(slots=True)
class LensEvalConfig:
    top_k: int = 5
    eps: float = 1e-12
    char_ngram: int = 3
    lowercase: bool = True
    strip_whitespace: bool = True


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)], clean_up_tokenization_spaces=False
    ).replace("\n", "\\n")


def _normalize_token_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_whitespace: bool = True,
) -> str:
    text = text.replace("▁", " ").replace("Ġ", " ").replace("Ċ", " ")
    if strip_whitespace:
        text = " ".join(text.split())
    if lowercase:
        text = text.lower()
    return text


def _char_ngrams(text: str, n: int) -> set[str]:
    if not text:
        return set()
    if n <= 1:
        return set(text)
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def evaluate_surface_metrics(
    lens_logits: torch.Tensor,
    final_logits: torch.Tensor,
    gold_token_id: int,
    *,
    tokenizer,
    config: SurfaceFormConfig | LensEvalConfig | None = None,
) -> LensSurfaceMetrics:
    """Surface-form proxies for gist-like agreement.

    These are still token-local, but more forgiving than exact-match metrics.
    """

    cfg = config or SurfaceFormConfig()
    k = max(1, int(cfg.top_k))
    n = max(1, int(cfg.char_ngram))

    lens_topk = lens_logits.topk(k).indices.tolist()
    final_topk = final_logits.topk(k).indices.tolist()
    lens_top1 = lens_topk[0]
    final_top1 = final_topk[0]

    lens_topk_texts = [_decode_token(tokenizer, t) for t in lens_topk]
    final_top1_text = _decode_token(tokenizer, final_top1)
    gold_text = _decode_token(tokenizer, gold_token_id)
    lens_top1_text = lens_topk_texts[0]

    def norm(text: str) -> str:
        return _normalize_token_text(
            text,
            lowercase=cfg.lowercase,
            strip_whitespace=cfg.strip_whitespace,
        )

    gold_norm = norm(gold_text)
    final_top1_norm = norm(final_top1_text)
    lens_top1_norm = norm(lens_top1_text)
    lens_topk_norms = [norm(t) for t in lens_topk_texts]

    gold_ngrams = _char_ngrams(gold_norm, n)
    final_ngrams = _char_ngrams(final_top1_norm, n)
    top1_ngrams = _char_ngrams(lens_top1_norm, n)

    best_vs_gold = 0.0
    best_vs_final = 0.0
    for candidate in lens_topk_norms:
        candidate_ngrams = _char_ngrams(candidate, n)
        best_vs_gold = max(best_vs_gold, _jaccard(candidate_ngrams, gold_ngrams))
        best_vs_final = max(best_vs_final, _jaccard(candidate_ngrams, final_ngrams))

    return LensSurfaceMetrics(
        top1_norm_matches_gold=(lens_top1_norm == gold_norm),
        top1_norm_matches_final_top1=(lens_top1_norm == final_top1_norm),
        topk_contains_norm_gold=(gold_norm in lens_topk_norms),
        top1_charngram_jaccard_gold=_jaccard(top1_ngrams, gold_ngrams),
        top1_charngram_jaccard_final_top1=_jaccard(top1_ngrams, final_ngrams),
        best_topk_charngram_jaccard_gold=best_vs_gold,
        best_topk_charngram_jaccard_final_top1=best_vs_final,
    )


def evaluate_residual_alignment(
    lens_residual: torch.Tensor,
    final_residual: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> ResidualAlignmentMetrics:
    """Compare a lens-mapped residual to the actual final residual.

    This is a simple mechanistic-faithfulness proxy in hidden space rather than
    vocab space.
    """

    lens_vec = lens_residual.float().reshape(-1)
    final_vec = final_residual.float().reshape(-1)

    l2 = torch.norm(lens_vec - final_vec, p=2)
    final_norm = torch.norm(final_vec, p=2)
    lens_norm = torch.norm(lens_vec, p=2)
    cosine = F.cosine_similarity(lens_vec.unsqueeze(0), final_vec.unsqueeze(0), dim=1)

    return ResidualAlignmentMetrics(
        cosine_to_final=float(cosine.item()),
        l2_to_final=float(l2.item()),
        relative_l2_to_final=float((l2 / (final_norm + eps)).item()),
        norm_ratio_to_final=float((lens_norm / (final_norm + eps)).item()),
    )
