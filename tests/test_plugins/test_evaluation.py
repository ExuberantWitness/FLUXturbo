"""Test NoveltyEvaluator with fixtures."""

import json
import os
from unittest.mock import patch

import numpy as np

from ccchain.core.ontology import Atom, Edge, Trajectory
from ccchain.plugins.evaluation import NoveltyEvaluator, _hausdorff

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def test_hausdorff_same_sets():
    """Hausdorff distance between identical sets should be near 0."""
    a = np.random.randn(5, 1024).astype(np.float32)
    d = _hausdorff(a, a)
    assert d < 1e-5


def test_hausdorff_empty():
    """Hausdorff with one empty set should be 1.0."""
    a = np.random.randn(3, 1024).astype(np.float32)
    b = np.empty((0, 1024), dtype=np.float32)
    d = _hausdorff(a, b)
    assert d == 1.0


def test_hausdorff_both_empty():
    """Hausdorff with both empty should be 0.0."""
    a = np.empty((0, 1024), dtype=np.float32)
    b = np.empty((0, 1024), dtype=np.float32)
    d = _hausdorff(a, b)
    assert d == 0.0


def test_hausdorff_different_sets():
    """Hausdorff between different sets should be > 0."""
    a = np.ones((3, 1024), dtype=np.float32)
    b = -np.ones((3, 1024), dtype=np.float32)
    d = _hausdorff(a, b)
    assert d > 0.5  # cosine distance between (1,...,1) and (-1,...,-1) ≈ 2


@patch("ccchain.core.llm.chat_json")
def test_evaluate_with_existing_trajectories(mock_chat):
    """Test full evaluation pipeline produces expected report structure."""
    mock_chat.return_value = _load_fixture("rubric_response.json")

    # Build proposal atoms
    proposal_atoms = [
        Atom(node_id="p_w2", name="Proposal Problem", type="bottleneck",
             level="W2_problem_analysis", context="Test problem",
             embedding=np.random.randn(1024).astype(np.float32)),
        Atom(node_id="p_w3", name="Proposal Direction", type="method",
             level="W3_solution_direction", context="Test direction",
             embedding=np.random.randn(1024).astype(np.float32)),
    ]
    proposal_edges: list[Edge] = []

    # Build existing trajectory
    existing = Trajectory(
        W2_problem=Atom(node_id="e_w2", name="Existing Problem", type="bottleneck",
                        level="W2_problem_analysis", context="Old problem",
                        embedding=np.random.randn(1024).astype(np.float32)),
        W3_solutions=[Atom(node_id="e_w3", name="Existing Direction", type="method",
                           level="W3_solution_direction", context="Old direction",
                           embedding=np.random.randn(1024).astype(np.float32))],
        source_pdf="old_paper.pdf",
    )

    evaluator = NoveltyEvaluator(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
    )

    report = evaluator.evaluate(proposal_atoms, proposal_edges, [existing])

    assert "novelty_score" in report
    assert "most_similar_trajectory" in report
    assert "level_distances" in report
    assert "divergence_points" in report
    assert "dimension_scores" in report
    assert "recommendation" in report
    assert 0.0 <= report["novelty_score"] <= 1.0


def test_evaluate_empty_trajectories():
    """Evaluate with no existing trajectories should return max novelty."""
    evaluator = NoveltyEvaluator(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
    )

    report = evaluator.evaluate([], [], [])
    assert report["novelty_score"] == 1.0
    assert report["most_similar_trajectory"] is None
