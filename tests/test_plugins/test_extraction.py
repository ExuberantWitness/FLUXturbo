"""Test TwoPhaseExtractor with JSON fixtures."""

import json
import os
from unittest.mock import patch

import pytest

from ccchain.core.ontology import Atom, Edge, Rho
from ccchain.plugins.extraction import TwoPhaseExtractor

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


@patch("ccchain.core.llm.chat_json")
def test_extract_phase1_parse(mock_chat):
    """Test that Phase 1 parse correctly transforms fixture JSON into atoms/edges."""
    mock_chat.return_value = _load_fixture("extract_w2w3.json")

    extractor = TwoPhaseExtractor(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
    )
    atoms, edges = extractor._extract_phase1("test text", "test.pdf")

    w2_atoms = [a for a in atoms if a.level == "W2_problem_analysis"]
    w3_atoms = [a for a in atoms if a.level == "W3_solution_direction"]

    assert len(w2_atoms) == 1
    assert len(w3_atoms) >= 1
    assert w2_atoms[0].type == "bottleneck"
    assert all(e.relation == "decomposes_into" or e.relation == "compares" for e in edges)


@patch("ccchain.core.llm.chat_json")
def test_extract_phase2_parse(mock_chat):
    """Test that Phase 2 parse correctly produces W4+W5 atoms."""
    mock_chat.return_value = _load_fixture("extract_w4w5.json")

    extractor = TwoPhaseExtractor(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
    )
    atoms, edges = extractor._extract_phase2("test text", "test.pdf", [])

    w4_atoms = [a for a in atoms if a.level == "W4_concrete_solution"]
    w5_atoms = [a for a in atoms if a.level == "W5_code_implementation"]

    assert len(w4_atoms) >= 1
    assert len(w5_atoms) >= 1
    assert w4_atoms[0].type == "solution"
    assert w5_atoms[0].type == "component"

    # Check that strong-causal edges have rho
    strong_edges = [e for e in edges if e.relation in ("extends", "improves")]
    for e in strong_edges:
        assert e.rho is not None, f"Edge {e.relation} missing rho"
        assert e.rho.confidence > 0


def test_atom_id_generation():
    """Test that atom IDs are generated with expected format."""
    extractor = TwoPhaseExtractor(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
    )
    aid = extractor._id("W4", "Sinkhorn OT", "paper.pdf")
    assert aid.startswith("W4_sinkhorn_ot_paper_")
    assert len(aid) > 20
