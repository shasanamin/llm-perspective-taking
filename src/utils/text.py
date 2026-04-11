from __future__ import annotations

import re


THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
BARE_NUMBER_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[.,!?;:]*\s*$")
TRAILING_PUNCTUATION_PATTERN = re.compile(r"[.,!?;:]*\s*$")


def strip_thinking(text: str) -> str:
    return THINK_TAG_PATTERN.sub("", text).strip()


def extract_thinking(text: str) -> str:
    match = THINK_TAG_PATTERN.search(text)
    if match is None:
        return ""
    return match.group(1).strip()


def _normalize_percentage_value(value: float, *, from_fraction: bool = False) -> float | None:
    if from_fraction:
        if not 0 <= value <= 1:
            return None
        value *= 100.0
    elif not 0 <= value <= 100:
        return None

    if value.is_integer():
        return int(value)
    return round(value, 4)


def extract_percentage(response_text: str) -> tuple[str, float | int | None]:
    text = strip_thinking(response_text)
    lowered = response_text.lower()
    if "i'm sorry" in lowered or "i am sorry" in lowered:
        return text, None

    matches = list(PERCENT_PATTERN.finditer(text))
    if matches:
        match = matches[-1]
        values = [_normalize_percentage_value(float(item.group(1))) for item in matches]
        if any(value is None for value in values):
            return text, None
        percentage = values[-1]

        if len(matches) > 1:
            unique_values = set(values)
            if len(unique_values) == 1:
                return text, percentage
            tail = text[match.end():]
            if TRAILING_PUNCTUATION_PATTERN.fullmatch(tail) is None:
                return text, None

        return text, percentage

    bare = BARE_NUMBER_PATTERN.fullmatch(text)
    if bare is None:
        return text, None

    value = float(bare.group(1))
    normalized = _normalize_percentage_value(value)
    if normalized is not None:
        return text, normalized

    normalized_fraction = _normalize_percentage_value(value, from_fraction=True)
    return text, normalized_fraction
