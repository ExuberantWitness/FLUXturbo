"""LLM calling utilities — single entry point for all LLM interactions."""

from __future__ import annotations

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
    import json

    return json.loads(response.choices[0].message.content or "{}")
