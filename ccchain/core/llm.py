"""LLM calling utilities — single entry point for all LLM interactions."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from openai import OpenAI


def get_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def chat(
    messages: list[dict],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Send a chat completion request, return the text response."""
    client = get_client(base_url, api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def chat_json(
    messages: list[dict],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict:
    """Send a chat completion request with JSON mode, return parsed dict."""
    client = get_client(base_url, api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")


def chat_json_majority(
    messages: list[dict],
    *,
    base_url: str,
    api_key: str,
    model: str,
    k: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> dict:
    """Run k independent chat_json calls; return majority-voted merged dict.

    Per-field voting:
      - Scalar fields (str/int/float/bool/None): pick the modal value across k samples.
        For floats, ties broken by first observation. For None vs value, value wins.
      - Dict fields: recurse one level — vote per sub-key.
      - List fields: pick the list whose length is modal; tie-break by first observation.

    Returns the merged dict. If all k calls fail or yield empty dicts, returns {}.
    """
    samples: list[dict] = []
    for _ in range(k):
        try:
            d = chat_json(
                messages, base_url=base_url, api_key=api_key, model=model,
                temperature=temperature, max_tokens=max_tokens,
            )
            if d:
                samples.append(d)
        except Exception:
            continue
    if not samples:
        return {}
    return _vote_merge(samples)


def _vote_merge(samples: list[dict]) -> dict:
    """Merge a list of dicts by per-key majority vote."""
    if not samples:
        return {}

    all_keys: set[str] = set()
    for s in samples:
        all_keys.update(s.keys())

    merged: dict[str, Any] = {}
    for key in all_keys:
        values = [s.get(key) for s in samples if key in s]
        if not values:
            continue
        merged[key] = _vote_value(values)
    return merged


def _vote_value(values: list[Any]) -> Any:
    """Vote on a single field's value across observations."""
    # Dict values — recurse
    if all(isinstance(v, dict) for v in values if v is not None):
        non_none = [v for v in values if v is not None]
        if not non_none:
            return None
        return _vote_merge(non_none)

    # List values — pick list with modal length
    if all(isinstance(v, list) for v in values if v is not None):
        non_none = [v for v in values if v is not None]
        if not non_none:
            return None
        lengths = Counter(len(v) for v in non_none)
        modal_len = lengths.most_common(1)[0][0]
        for v in non_none:
            if len(v) == modal_len:
                return v
        return non_none[0]

    # Scalar values — pick modal; None loses to any concrete value
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None
    counts = Counter(_hash_scalar(v) for v in non_none)
    modal_hash = counts.most_common(1)[0][0]
    for v in non_none:
        if _hash_scalar(v) == modal_hash:
            return v
    return non_none[0]


def _hash_scalar(v: Any) -> str:
    """Hashable representation of a scalar for voting."""
    if isinstance(v, float):
        # round floats to absorb minor numeric drift
        return f"f:{round(v, 4)}"
    if isinstance(v, bool):
        return f"b:{v}"
    if isinstance(v, int):
        return f"i:{v}"
    return f"s:{v}"

