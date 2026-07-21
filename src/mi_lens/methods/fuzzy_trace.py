from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class FuzzyTraceConfig:
    """Configuration for first-pass fuzzy-trace evaluation.

    This starter version focuses on token-level verbatim and
    distributional comparisons. Semantic/gist scoring can be layered on top
    later from decoded short continuations.
    """

    top_k: int = 5
    eps: float = 1e-12


@dataclass(slots=True)
class LensTokenMetrics:
    top1_exact: bool
    topk_exact: bool
    gold_rank: int
    gold_prob: float
    final_top1_rank: int
    final_top1_prob: float
    topk_jaccard_vs_final: float
    kl_vs_final: float
    js_vs_final: float
    tv_vs_final: float


def _topk_ids(logits: torch.Tensor, k: int) -> list[int]:
    return logits.topk(k).indices.tolist()


def _rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    return int((logits.argsort(descending=True) == token_id).nonzero()[0].item() + 1)


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=0)


def _jaccard(ids_a: list[int], ids_b: list[int]) -> float:
    a, b = set(ids_a), set(ids_b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float) -> float:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return float((p * (p.log() - q.log())).sum().item())


def _js_div(p: torch.Tensor, q: torch.Tensor, eps: float) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m, eps) + 0.5 * _kl_div(q, m, eps)


def _tv_dist(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * torch.abs(p - q).sum().item())


def evaluate_token_metrics(
    lens_logits: torch.Tensor,
    final_logits: torch.Tensor,
    gold_token_id: int,
    *,
    config: FuzzyTraceConfig | None = None,
) -> LensTokenMetrics:
    """Compare one lens logit vector against the final-model logit vector.

    Args:
        lens_logits: `[vocab_size]` logits from JL/logit lens/tuned lens.
        final_logits: `[vocab_size]` logits from the final model.
        gold_token_id: The true next-token id.
        config: Scoring configuration.
    """

    cfg = config or FuzzyTraceConfig()
    k = max(1, int(cfg.top_k))

    lens_topk = _topk_ids(lens_logits, k)
    final_topk = _topk_ids(final_logits, k)
    final_top1 = final_topk[0]

    p_lens = _softmax_probs(lens_logits)
    p_final = _softmax_probs(final_logits)

    return LensTokenMetrics(
        top1_exact=(lens_topk[0] == gold_token_id),
        topk_exact=(gold_token_id in lens_topk),
        gold_rank=_rank_of_token(lens_logits, gold_token_id),
        gold_prob=float(p_lens[gold_token_id].item()),
        final_top1_rank=_rank_of_token(lens_logits, final_top1),
        final_top1_prob=float(p_lens[final_top1].item()),
        topk_jaccard_vs_final=_jaccard(lens_topk, final_topk),
        kl_vs_final=_kl_div(p_lens, p_final, cfg.eps),
        js_vs_final=_js_div(p_lens, p_final, cfg.eps),
        tv_vs_final=_tv_dist(p_lens, p_final),
    )
