"""Test GraphRetriever with mock embeddings and graph."""

import igraph as ig
import numpy as np

from ccchain.plugins.retrieval import GraphRetriever


def _make_test_graph_and_index() -> tuple[ig.Graph, dict[str, int]]:
    """Build a test graph with W2-W5 atoms and return (graph, node_index)."""
    g = ig.Graph()
    g.add_vertex(name="w2_1", label="Problem X", type="bottleneck", level="W2_problem_analysis")
    g.add_vertex(name="w3_1", label="Solution A", type="method", level="W3_solution_direction")
    g.add_vertex(name="w4_1", label="Sinkhorn OT", type="method", level="W4_concrete_solution")
    g.add_vertex(name="w5_1", label="sinkhorn()", type="component", level="W5_code_implementation")
    g.add_vertex(name="w4_2", label="Shapley Credit", type="method", level="W4_concrete_solution")

    g.add_edge(0, 1, relation="decomposes_into")
    g.add_edge(1, 2, relation="decomposes_into")
    g.add_edge(2, 3, relation="decomposes_into")
    g.add_edge(3, 2, relation="aggregates_to")
    g.add_edge(2, 4, relation="extends")

    node_index = {v["name"]: v.index for v in g.vs}
    return g, node_index


def test_search_returns_filtered_results():
    """Test that search returns atoms at the requested level."""
    g, node_index = _make_test_graph_and_index()
    n = g.vcount()

    # Mock embeddings: W4 atoms get high similarity to "OT credit"
    embeddings = np.random.randn(n, 1024).astype(np.float32)
    # Make W4_1 (index 2) very similar to the query
    query_embedding = embeddings[2] + 0.01 * np.random.randn(1024).astype(np.float32)

    retriever = GraphRetriever(
        embedder_base_url="http://localhost:11434/v1",
        embedder_model="bge-m3:latest",
    )

    # Override embed to return our fixed embedding
    original_embed = __import__("ccchain.core.embedding", fromlist=["embed"]).embed

    class _MockEmbed:
        def __call__(self, texts, **kwargs):
            instruction = kwargs.get("instruction", "")
            if "atom" in instruction:
                return np.array([query_embedding])
            return np.array([query_embedding])

    import ccchain.plugins.retrieval as rmod
    rmod.embed = _MockEmbed()

    try:
        results = retriever.search(
            query="optimal transport credit assignment",
            top_k=3,
            level="W4",
            graph=g,
            embeddings=embeddings,
            node_index=node_index,
        )
    finally:
        rmod.embed = original_embed

    assert len(results) <= 3
    for r in results:
        assert r["level"] == "W4_concrete_solution"


def test_search_empty_graph():
    """Test search on an empty graph returns empty list."""
    g = ig.Graph()
    embeddings = np.empty((0, 1024), dtype=np.float32)
    node_index: dict[str, int] = {}

    retriever = GraphRetriever(
        embedder_base_url="http://localhost:11434/v1",
        embedder_model="bge-m3:latest",
    )

    # Mock embed
    import ccchain.plugins.retrieval as rmod
    original_embed = rmod.embed

    class _MockEmbed:
        def __call__(self, texts, **kwargs):
            return np.zeros((len(texts), 1024), dtype=np.float32)

    rmod.embed = _MockEmbed()

    try:
        results = retriever.search(
            query="test",
            top_k=10,
            level="W4",
            graph=g,
            embeddings=embeddings,
            node_index=node_index,
        )
    finally:
        rmod.embed = original_embed

    assert results == []
