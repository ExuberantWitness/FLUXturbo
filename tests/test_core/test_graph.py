"""Test igraph operations."""

import igraph as ig
import numpy as np
from ccchain.core.graph import (
    ppr,
    connected_components_by_level,
    upward_trace,
    downward_expand,
    local_ppr_path,
)


def _make_test_graph() -> ig.Graph:
    """Build a small test graph with W2-W5 levels."""
    g = ig.Graph(directed=True)
    g.add_vertices(6)
    g.vs[0].update_attributes(name="w2", label="Problem", type="bottleneck", level="W1_problem")
    g.vs[1].update_attributes(name="w3a", label="Direction A", type="method", level="W2_direction")
    g.vs[2].update_attributes(name="w3b", label="Direction B", type="method", level="W2_direction")
    g.vs[3].update_attributes(name="w4a", label="Solution A1", type="method", level="W4_implementation")
    g.vs[4].update_attributes(name="w4b", label="Solution B1", type="method", level="W4_implementation")
    g.vs[5].update_attributes(name="w5a", label="Code A1", type="component", level="W5_code")

    g.add_edges(
        [(0, 1), (0, 2)],  # W2 → W3
        {"relation": "decomposes_into", "weight": 1.0},
    )
    g.add_edges(
        [(1, 3), (2, 4)],  # W3 → W4
        {"relation": "decomposes_into", "weight": 1.0},
    )
    g.add_edges(
        [(3, 5)],  # W4 → W5
        {"relation": "decomposes_into", "weight": 1.0},
    )
    # Cross-edges: extends
    g.add_edge(3, 4, relation="extends", weight=1.0)
    # Reverse: aggregates_to
    g.add_edge(5, 3, relation="aggregates_to", weight=1.0)
    g.add_edge(3, 1, relation="aggregates_to", weight=1.0)

    return g


def test_ppr_basic():
    g = ig.Graph()
    g.add_vertices(4)
    g.add_edges([(0, 1), (1, 2), (2, 3)])
    reset = np.zeros(4)
    reset[0] = 1.0
    scores = ppr(g, reset)
    assert scores.shape == (4,)
    assert scores[0] > scores[3]


def test_connected_components_by_level():
    g = _make_test_graph()
    comps = connected_components_by_level(g, "W2_direction")
    assert len(comps) > 0


def test_upward_trace():
    g = _make_test_graph()
    path = upward_trace(g, start_idx=5)  # start from W5
    assert len(path) >= 1
    assert path[0] == 5


def test_downward_expand():
    g = _make_test_graph()
    reachable = downward_expand(g, start_idx=0)  # start from W2
    assert len(reachable) >= 2


def test_local_ppr_path():
    g = _make_test_graph()
    # Seed from W4(v3), candidates are W3(v1, v2)
    best = local_ppr_path(g, seed_idx=3, candidate_indices=[1, 2])
    assert best in [1, 2]


def test_ppr_damping():
    g = ig.Graph()
    g.add_vertices(3)
    g.add_edges([(0, 1), (1, 2)])
    reset = np.array([1.0, 0.0, 0.0])
    scores = ppr(g, reset, damping=0.5)
    assert scores[0] > 0.4  # seed should retain high score
