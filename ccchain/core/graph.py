"""igraph operations: PPR, connected components, level traversal."""

from __future__ import annotations

import numpy as np
import igraph as ig


def ppr(
    graph: ig.Graph,
    reset_vector: np.ndarray,
    damping: float = 0.5,
    implementation: str = "prpack",
) -> np.ndarray:
    """Personalized PageRank over the graph.

    Args:
        graph: igraph Graph object.
        reset_vector: (n,) personalization vector (sums to 1).
        damping: Damping factor (1 - teleport probability).
        implementation: igraph PPR implementation ("prpack" or "power").

    Returns:
        (n,) PPR score vector.
    """
    reset = np.asarray(reset_vector, dtype=np.float64)
    reset = reset / reset.sum()
    result = graph.personalized_pagerank(
        reset=reset.tolist(),
        damping=damping,
        implementation=implementation,
    )
    return np.array(result, dtype=np.float64)


def connected_components_by_level(
    graph: ig.Graph,
    level: str,
) -> list[list[int]]:
    """Find connected components among vertices of a given level.

    Only considers edges between same-level vertices (horizontal edges).
    Returns list of vertex index groups.
    """
    level_verts = [v.index for v in graph.vs if v["level"] == level]
    if len(level_verts) <= 1:
        return [level_verts] if level_verts else []

    subgraph = graph.subgraph(level_verts)
    comps = subgraph.connected_components()
    return [[level_verts[i] for i in comp] for comp in comps]


def upward_trace(
    graph: ig.Graph,
    start_idx: int,
    target_level: str | None = None,
) -> list[int]:
    """Trace upward along AGGREGATES_TO edges from a vertex.

    Args:
        graph: igraph Graph.
        start_idx: Starting vertex index.
        target_level: Optional target level to stop at.

    Returns:
        Ordered list of vertex indices along the upward path.
    """
    path: list[int] = [start_idx]
    current = start_idx

    while True:
        out_edges = graph.incident(current, mode="out")
        upward_neighbors: list[int] = []
        for eid in out_edges:
            if graph.es[eid]["relation"] == "aggregates_to":
                tgt = graph.es[eid].target
                if tgt != current:
                    upward_neighbors.append(tgt)

        if not upward_neighbors:
            break

        next_v = upward_neighbors[0]  # take highest-weight edge
        path.append(next_v)
        current = next_v

        if target_level and graph.vs[current]["level"] == target_level:
            break

    return path


def downward_expand(
    graph: ig.Graph,
    start_idx: int,
    target_level: str | None = None,
) -> list[int]:
    """Expand downward along DECOMPOSES_INTO edges from a vertex.

    Args:
        graph: igraph Graph.
        start_idx: Starting vertex index.
        target_level: Optional target level to stop at.

    Returns:
        List of reachable vertex indices.
    """
    visited: set[int] = set()
    stack: list[int] = [start_idx]

    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)

        if target_level and graph.vs[v]["level"] == target_level:
            continue

        out_edges = graph.incident(v, mode="out")
        for eid in out_edges:
            if graph.es[eid]["relation"] == "decomposes_into":
                tgt = graph.es[eid].target
                if tgt not in visited:
                    stack.append(tgt)

    return list(visited)


def local_ppr_path(
    graph: ig.Graph,
    seed_idx: int,
    candidate_indices: list[int],
) -> int:
    """Pick the best ancestor from candidates via local PPR from seed.

    Args:
        graph: igraph Graph.
        seed_idx: Starting vertex index.
        candidate_indices: Candidate vertex indices to choose from.

    Returns:
        The candidate index with highest PPR score.
    """
    if not candidate_indices:
        return -1
    if len(candidate_indices) == 1:
        return candidate_indices[0]

    n = graph.vcount()
    reset = np.zeros(n)
    reset[seed_idx] = 1.0

    scores = ppr(graph, reset)
    best = max(candidate_indices, key=lambda i: scores[i])
    return best
