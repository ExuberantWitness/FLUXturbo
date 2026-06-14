"""BGE-M3 embedding via Ollama/OpenAI-compatible endpoint."""

from __future__ import annotations

import numpy as np
from openai import OpenAI


def embed(
    texts: list[str],
    *,
    base_url: str,
    model: str,
    api_key: str = "ollama",
    instruction: str | None = None,
) -> np.ndarray:
    """Generate BGE-M3 embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        base_url: Embedding service base URL (e.g. http://localhost:11434/v1).
        model: Embedding model name (e.g. bge-m3:latest).
        api_key: API key for the service.
        instruction: Optional BGE-M3 instruction prefix
            (e.g. 'Represent this query for atom retrieval').

    Returns:
        (n, 1024) float32 numpy array.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)

    inputs = texts
    if instruction is not None:
        inputs = [f"{instruction}: {t}" for t in texts]

    response = client.embeddings.create(model=model, input=inputs)
    embeddings = [d.embedding for d in response.data]
    return np.array(embeddings, dtype=np.float32)
