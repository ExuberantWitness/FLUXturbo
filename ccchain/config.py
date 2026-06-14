"""Configuration management for ccchain."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Storage ---
    db_path: str = "blueprint_output/cc_base.db"
    graph_dir: str = "blueprint_output/"

    # --- LLM (extraction + reduction + evaluation) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:latest"

    # --- Embedding ---
    embedder_base_url: str = "http://localhost:11434/v1"
    embedder_model: str = "bge-m3:latest"

    # --- Retrieval parameters ---
    link_top_k: int = 10
    ppr_damping: float = 0.5
    w5_bias_weight: float = 0.05

    # --- Refiner ---
    max_refine_rounds: int = 3

    # --- Evaluation ---
    hausdorff_weights: dict[str, float] = field(default_factory=lambda: {
        "W2": 0.4,
        "W3": 0.3,
        "W4": 0.2,
        "W5": 0.1,
    })
