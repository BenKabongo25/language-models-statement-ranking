from __future__ import annotations

import re
from typing import Any

STAGE1_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_span", "statement", "type"],
        "properties": {
            "source_span": {"type": "string", "minLength": 1},
            "statement": {"type": "string", "minLength": 1},
            "type": {"enum": ["user", "item", "interaction", "external"]},
        },
    },
}

STAGE2_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["user", "item"],
    "properties": {
        "user": {"type": "string", "minLength": 1},
        "item": {"type": "string", "minLength": 1},
    },
}

STAGE3_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stance"],
    "properties": {"stance": {"enum": ["positive", "negative", "neutral"]}},
}

_MONEY = re.compile(
    r"(?:[$€£¥]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:usd|eur|gbp|dollars?|euros?|pounds?|yen)\b)",
    flags=re.IGNORECASE,
)


def _contains_money(text: str) -> bool:
    return bool(_MONEY.search(text))


def _contains_forbidden_brand(text: str, forbidden_brand: str | None) -> bool:
    brand = (forbidden_brand or "").strip()
    if len(brand) < 2:
        return False
    return brand.casefold() in text.casefold()


def validate_stage1(
    value: Any,
    *,
    review: str,
    review_title: str | None,
    forbidden_brand: str | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    if not isinstance(value, list):
        return [], ["Stage 1 output must be a JSON array."]

    valid: list[dict[str, str]] = []
    errors: list[str] = []
    allowed_types = {"user", "item", "interaction", "external"}
    sources = [review]
    if review_title is not None:
        sources.append(review_title)

    for index, obj in enumerate(value):
        prefix = f"statement[{index}]"
        if not isinstance(obj, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        if set(obj) != {"source_span", "statement", "type"}:
            errors.append(f"{prefix} must contain exactly source_span, statement, and type.")
            continue
        span = obj.get("source_span")
        statement = obj.get("statement")
        kind = obj.get("type")
        local_errors: list[str] = []
        if not isinstance(span, str) or not span.strip():
            local_errors.append("source_span must be a non-empty string")
        elif not any(span in source for source in sources):
            local_errors.append("source_span is not an exact contiguous span of REVIEW_TITLE or REVIEW")
        if not isinstance(statement, str) or not statement.strip():
            local_errors.append("statement must be a non-empty string")
        else:
            if _contains_money(statement):
                local_errors.append("normalized statement contains an exact monetary amount")
            if _contains_forbidden_brand(statement, forbidden_brand):
                local_errors.append("normalized statement contains the reviewed item's brand name")
        if kind not in allowed_types:
            local_errors.append(f"type must be one of {sorted(allowed_types)}")
        if local_errors:
            errors.extend(f"{prefix}: {error}." for error in local_errors)
        else:
            valid.append({"source_span": span.strip(), "statement": statement.strip(), "type": str(kind)})
    return valid, errors


def validate_stage2(
    value: Any,
    *,
    forbidden_brand: str | None = None,
) -> tuple[dict[str, str] | None, list[str]]:
    if not isinstance(value, dict) or set(value) != {"user", "item"}:
        return None, ["Stage 2 output must contain exactly user and item."]
    errors: list[str] = []
    cleaned: dict[str, str] = {}
    for key in ("user", "item"):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{key} must be a non-empty string.")
            continue
        text = text.strip()
        if _contains_money(text):
            errors.append(f"{key} contains an exact monetary amount.")
        if _contains_forbidden_brand(text, forbidden_brand):
            errors.append(f"{key} contains the reviewed item's brand name.")
        cleaned[key] = text
    return (cleaned if not errors else None), errors


def validate_stage3(value: Any) -> tuple[dict[str, str] | None, list[str]]:
    if not isinstance(value, dict) or set(value) != {"stance"}:
        return None, ["Stage 3 output must contain exactly stance."]
    stance = value.get("stance")
    if stance not in {"positive", "negative", "neutral"}:
        return None, ["stance must be positive, negative, or neutral."]
    return {"stance": str(stance)}, []
