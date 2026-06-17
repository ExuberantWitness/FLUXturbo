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
        "W1": 0.15,   # problem
        "W2": 0.25,   # direction
        "W3": 0.30,   # approach/思路 — highest discriminative weight
        "W4": 0.20,   # implementation
        "W5": 0.10,   # code
    })

    # --- Reference resolution (I3) ---
    reference_api_timeout: float = 2.0
    reference_api_max_retries: int = 3
    reference_api_keys: dict[str, str] = field(default_factory=dict)
    # keys: "semantic_scholar", "openalex", "crossref"

    # --- CoE Audit ---
    audit_majority_k: int = 5
    audit_i1_k: int = 1

    # --- Cross-paper hierarchical consolidation (auto-runs in ingest) ---
    auto_consolidate: bool = True
    consolidate_similarity_threshold: float = 0.85
    consolidate_majority_k: int = 3
