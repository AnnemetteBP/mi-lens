from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any


WORD_RE = re.compile(r"\S+")


@dataclass(slots=True)
class TokenAuditConfig:
    budgets: tuple[int, ...] = (128, 256, 512)
    text_keys: tuple[str, ...] = ("prompt", "text", "question")
    add_special_tokens: bool = True


@dataclass(slots=True)
class TokenAuditRow:
    example_id: str
    model_label: str
    language: str
    text: str
    char_count: int
    word_count: int
    token_count: int
    chars_per_token: float
    words_per_token: float
    tokenization_fragmentation_ratio: float
    token_ids: list[int]
    tokens: list[str]
    fits_budget: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_text(record: Mapping[str, Any], text_keys: Sequence[str]) -> str:
    for key in text_keys:
        value = record.get(key)
        if value is not None:
            return str(value)
    raise KeyError(
        f"Could not find prompt text in record. Tried keys: {list(text_keys)!r}"
    )


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def build_token_audit_rows(
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    language: str,
    config: TokenAuditConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or TokenAuditConfig()
    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        example_id = str(record.get("id", idx))
        text = _get_text(record, cfg.text_keys)
        encoding = tokenizer(
            text,
            add_special_tokens=cfg.add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_ids = [int(token_id) for token_id in encoding["input_ids"]]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        char_count = len(text)
        word_count = _word_count(text)
        token_count = len(token_ids)
        row = TokenAuditRow(
            example_id=example_id,
            model_label=model_label,
            language=language,
            text=text,
            char_count=char_count,
            word_count=word_count,
            token_count=token_count,
            chars_per_token=_safe_div(char_count, token_count),
            words_per_token=_safe_div(word_count, token_count),
            tokenization_fragmentation_ratio=_safe_div(token_count, word_count),
            token_ids=token_ids,
            tokens=tokens,
            fits_budget={
                str(int(budget)): token_count <= int(budget) for budget in cfg.budgets
            },
        )
        rows.append(row.to_dict())

    return rows


def pair_token_audit_rows(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    left_language: str,
    right_language: str,
) -> list[dict[str, Any]]:
    right_by_id = {str(row["example_id"]): row for row in right_rows}
    pair_rows: list[dict[str, Any]] = []

    for left_row in left_rows:
        example_id = str(left_row["example_id"])
        right_row = right_by_id.get(example_id)
        if right_row is None:
            continue

        left_tokens = int(left_row["token_count"])
        right_tokens = int(right_row["token_count"])
        shared_fit = {
            budget: bool(left_row["fits_budget"][budget] and right_row["fits_budget"][budget])
            for budget in left_row["fits_budget"].keys()
            if budget in right_row["fits_budget"]
        }

        pair_rows.append(
            {
                "example_id": example_id,
                "model_label": left_row["model_label"],
                "left_language": left_language,
                "right_language": right_language,
                "left_char_count": int(left_row["char_count"]),
                "right_char_count": int(right_row["char_count"]),
                "left_word_count": int(left_row["word_count"]),
                "right_word_count": int(right_row["word_count"]),
                "left_token_count": left_tokens,
                "right_token_count": right_tokens,
                "token_diff": left_tokens - right_tokens,
                "token_ratio": _safe_div(left_tokens, right_tokens),
                "fragmentation_ratio_diff": float(
                    left_row["tokenization_fragmentation_ratio"]
                    - right_row["tokenization_fragmentation_ratio"]
                ),
                "shared_fit_budget": shared_fit,
            }
        )

    return pair_rows


def summarize_token_audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0}

    token_counts = [int(row["token_count"]) for row in rows]
    chars_per_token = [float(row["chars_per_token"]) for row in rows]
    words_per_token = [float(row["words_per_token"]) for row in rows]
    fragmentation = [float(row["tokenization_fragmentation_ratio"]) for row in rows]

    budget_keys = list(rows[0]["fits_budget"].keys())
    fits_budget = {
        budget: _safe_div(
            sum(1 for row in rows if row["fits_budget"][budget]),
            len(rows),
        )
        for budget in budget_keys
    }

    return {
        "num_rows": len(rows),
        "token_count_mean": mean(token_counts),
        "token_count_std": pstdev(token_counts) if len(token_counts) > 1 else 0.0,
        "chars_per_token_mean": mean(chars_per_token),
        "words_per_token_mean": mean(words_per_token),
        "fragmentation_ratio_mean": mean(fragmentation),
        "fits_budget_fraction": fits_budget,
    }


def summarize_paired_token_audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0}

    token_diffs = [int(row["token_diff"]) for row in rows]
    token_ratios = [float(row["token_ratio"]) for row in rows]
    frag_diffs = [float(row["fragmentation_ratio_diff"]) for row in rows]
    budget_keys = list(rows[0]["shared_fit_budget"].keys())
    shared_fit = {
        budget: _safe_div(
            sum(1 for row in rows if row["shared_fit_budget"][budget]),
            len(rows),
        )
        for budget in budget_keys
    }

    return {
        "num_rows": len(rows),
        "token_diff_mean": mean(token_diffs),
        "token_diff_std": pstdev(token_diffs) if len(token_diffs) > 1 else 0.0,
        "token_ratio_mean": mean(token_ratios),
        "fragmentation_ratio_diff_mean": mean(frag_diffs),
        "shared_fit_fraction": shared_fit,
    }
