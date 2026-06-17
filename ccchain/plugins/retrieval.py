"""GraphRetriever — HippoRAG-style dual-instruction embedding + dense recall + PPR diffusion.

W5 atoms serve as passages. No LLM rerank — PPR signal is sufficient.
"""

from __future__ import annotations

import numpy as np
import igraph as ig

from ccchain.core.embedding import embed
from ccchain.core.graph import ppr
from ccchain.core.ontology import LEVEL_ALIAS
from ccchain.plugins.base import Retriever


class GraphRetriever(Retriever):
    """HippoRAG-pattern retrieval: dual embedding → dense recall → PPR diffusion."""

    def __init__(
        self,
        *,
        embedder_base_url: str,
        embedder_model: str,
        embedder_api_key: str = "ollama",
        link_top_k: int = 10,
        ppr_damping: float = 0.5,
        w5_bias_weight: float = 0.05,
    ):
        self.embedder_base_url = embedder_base_url
        self.embedder_model = embedder_model
        self.embedder_api_key = embedder_api_key
        self.link_top_k = link_top_k
        self.ppr_damping = ppr_damping
        self.w5_bias_weight = w5_bias_weight

    def search(
        self,
        query: str,
        top_k: int,
        level: str,
        graph: ig.Graph,
        embeddings: np.ndarray,
        node_index: dict[str, int],
    ) -> list[dict]:
        full_level = LEVEL_ALIAS.get(level, level)

        n = graph.vcount()
        if n == 0 or embeddings.shape[0] == 0:
            return []

        # 1. Dual instruction query embedding (HippoRAG pattern)
        q_atom = embed(
            [query],
            base_url=self.embedder_base_url,
            model=self.embedder_model,
            api_key=self.embedder_api_key,
            instruction="Represent this query for atom retrieval",
        )[0]

        q_w5 = embed(
            [query],
            base_url=self.embedder_base_url,
            model=self.embedder_model,
            api_key=self.embedder_api_key,
            instruction="Represent this query for code passage retrieval",
        )[0]

        # 2. Dense atom recall (all levels) → top link_top_k
        atom_scores = embeddings @ q_atom
        top_indices = np.argsort(atom_scores)[-self.link_top_k:]
        top_scores = atom_scores[top_indices]
        atom_weights = _softmax(top_scores)

        # 3. W5 dense recall → w5_bias tiebreaker
        w5_indices = [
            node_index[name] for name, idx in node_index.items()
            if graph.vs[idx]["level"] == "W5_code"
            and idx < len(embeddings)
        ]
        n = graph.vcount()
        w5_bias = np.zeros(n)

        if w5_indices:
            w5_embs = embeddings[w5_indices]
            w5_scores = w5_embs @ q_w5
            w5_probs = _softmax(w5_scores)
            for i, vi in enumerate(w5_indices):
                w5_bias[vi] = float(w5_probs[i]) * self.w5_bias_weight

        # 4. PPR: reset = atom_weights + w5_bias
        reset = np.zeros(n)
        for i, vi in enumerate(top_indices):
            reset[vi] = float(atom_weights[i])
        reset = reset + w5_bias
        total = reset.sum()
        if total > 0:
            reset = reset / total

        # Convert to undirected for PPR (HippoRAG pattern)
        undirected = graph.copy()
        undirected.to_undirected(combine_edges="ignore")

        ppr_scores = ppr(undirected, reset, damping=self.ppr_damping)

        # 5. Filter by level, sort, return top_k
        results: list[dict] = []
        for v in graph.vs:
            if v["level"] != full_level:
                continue
            vi = v.index
            results.append({
                "node_id": v["name"],
                "name": v["label"],
                "level": v["level"],
                "type": v["type"],
                "score": float(ppr_scores[vi]),
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    e = np.exp(x - x.max())
    return e / e.sum()
