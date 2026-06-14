"""Test HierarchicalReducer with JSON fixtures."""

import json
import os
from unittest.mock import patch

import igraph as ig

from ccchain.core.ontology import Atom
from ccchain.plugins.reduction import HierarchicalReducer

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _make_w5_atoms() -> list[Atom]:
    return [
        Atom(node_id="w5_1", name="sinkhorn_credit", type="component",
             level="W5_code_implementation", context="Sinkhorn OT credit assignment"),
        Atom(node_id="w5_2", name="wasserstein_credit", type="component",
             level="W5_code_implementation", context="Wasserstein-1 credit assignment"),
    ]


def _make_test_graph(atoms: list[Atom]) -> ig.Graph:
    g = ig.Graph(directed=True)
    for a in atoms:
        g.add_vertex(name=a.node_id, label=a.name, type=a.type, level=a.level)
    return g


@patch("ccchain.core.llm.chat_json")
def test_reduce_w5_to_w4(mock_chat):
    """Test that W5→W4 reduction produces higher-level atoms."""
    mock_chat.return_value = _load_fixture("reduce_w5_to_w4.json")

    reducer = HierarchicalReducer(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
    )
    atoms = _make_w5_atoms()
    graph = _make_test_graph(atoms)

    result = reducer.reduce_level(
        atoms=atoms,
        edges=[],
        from_level="W5_code_implementation",
        to_level="W4_concrete_solution",
        graph=graph,
    )

    assert len(result) >= 1
    assert result[0].level == "W4_concrete_solution"
    assert len(result[0].context) > 0


def test_reduce_empty_component():
    """Test that an empty graph returns no new atoms."""
    reducer = HierarchicalReducer(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
    )
    graph = ig.Graph(directed=True)

    result = reducer.reduce_level(
        atoms=[],
        edges=[],
        from_level="W5_code_implementation",
        to_level="W4_concrete_solution",
        graph=graph,
    )
    assert result == []
