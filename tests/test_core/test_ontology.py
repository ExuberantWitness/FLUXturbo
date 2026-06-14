"""Test ontology constants and dataclasses."""

import pytest
from ccchain.core.ontology import (
    ATOM_TYPES,
    ATOM_TYPE_SET,
    CC_EDGE_TYPES,
    HIERARCHY_EDGES,
    LEVELS,
    LEVEL_ORDER,
    LEVEL_ALIAS,
    STRONG_CAUSAL_EDGES,
    TYPE_COMPATIBILITY,
    Atom,
    Edge,
    Rho,
    Trajectory,
)


def test_atom_types_count():
    assert len(ATOM_TYPES) == 8


def test_levels_order():
    assert len(LEVELS) == 4
    assert LEVEL_ORDER["W2_problem_analysis"] == 0
    assert LEVEL_ORDER["W5_code_implementation"] == 3


def test_level_alias_mapping():
    assert LEVEL_ALIAS["W2"] == "W2_problem_analysis"
    assert LEVEL_ALIAS["W4"] == "W4_concrete_solution"


def test_strong_causal_edges():
    assert "extends" in STRONG_CAUSAL_EDGES
    assert "improves" in STRONG_CAUSAL_EDGES
    assert "related_to" not in STRONG_CAUSAL_EDGES


def test_hierarchy_edges():
    assert "aggregates_to" in HIERARCHY_EDGES
    assert "decomposes_into" in HIERARCHY_EDGES


def test_type_compatibility():
    assert ("method", "method") in TYPE_COMPATIBILITY["extends"]
    assert ("method", "component") in TYPE_COMPATIBILITY["uses_component"]


def test_atom_creation():
    a = Atom(
        node_id="W4_test_1",
        name="Test Atom",
        type="method",
        level="W4_concrete_solution",
        context="Test context",
    )
    assert a.node_id == "W4_test_1"
    assert a.type == "method"
    assert a.status == "active"


def test_atom_invalid_type():
    with pytest.raises(ValueError):
        Atom(node_id="X", name="Bad", type="invalid_type", level="W4_concrete_solution")


def test_atom_invalid_level():
    with pytest.raises(ValueError):
        Atom(node_id="X", name="Bad", type="method", level="W99")


def test_atom_to_from_dict():
    a = Atom(
        node_id="W4_test",
        name="Test",
        type="method",
        level="W4_concrete_solution",
        context="Ctx",
        tags=["domain:MARL"],
    )
    d = a.to_dict()
    assert d["node_id"] == "W4_test"
    a2 = Atom.from_dict(d)
    assert a2.name == "Test"
    assert a2.tags == ["domain:MARL"]


def test_edge_creation():
    rho = Rho(
        bottleneck="credit_assignment",
        mechanism="OT distance",
        tradeoff="Computational cost",
        confidence=0.9,
    )
    e = Edge(src="A", relation="extends", tgt="B", rho=rho)
    assert e.src == "A"
    assert e.rho.confidence == 0.9


def test_edge_to_from_dict():
    e = Edge(src="A", relation="extends", tgt="B", weight=0.5)
    d = e.to_dict()
    assert d["relation"] == "extends"
    e2 = Edge.from_dict(d)
    assert e2.weight == 0.5


def test_trajectory_empty():
    t = Trajectory(source_pdf="test.pdf")
    embs = t.get_embeddings_by_level()
    assert embs["W2"].shape == (0, 1024)
    assert embs["W4"].shape == (0, 1024)


def test_atom_v02_fields_default_none():
    """New v0.2 fields default to None when not provided."""
    a = Atom(
        node_id="W5_test",
        name="Sinkhorn",
        type="method",
        level="W5_code_implementation",
    )
    assert a.code_body is None
    assert a.source_refs is None
    assert a.provenance is None
    assert a.rowid is None


def test_atom_v02_fields_roundtrip():
    """code_body, source_refs, provenance survive to_dict/from_dict roundtrip.
    rowid is excluded from serialization (DB implementation detail)."""
    a = Atom(
        node_id="W5_test",
        name="Sinkhorn",
        type="method",
        level="W5_code_implementation",
        code_body="def sinkhorn2(cost, reg=0.1): ...",
        source_refs=["arxiv:2203.12345", "paper:OT-CTDE-2024"],
        provenance={"cycle": 1, "phase": "extract", "via": "TwoPhaseExtractor"},
        rowid=42,
    )
    d = a.to_dict()
    assert d["code_body"] == "def sinkhorn2(cost, reg=0.1): ..."
    assert d["source_refs"] == ["arxiv:2203.12345", "paper:OT-CTDE-2024"]
    assert d["provenance"]["cycle"] == 1
    assert "rowid" not in d  # rowid is not serialized

    a2 = Atom.from_dict(d)
    assert a2.code_body == a.code_body
    assert a2.source_refs == a.source_refs
    assert a2.provenance == a.provenance
    assert a2.rowid is None  # roundtrip from dict yields None


def test_edge_v02_provenance_roundtrip():
    """Edge.provenance survives to_dict/from_dict; rowid excluded from serialization."""
    e = Edge(
        src="A",
        relation="extends",
        tgt="B",
        provenance={"cycle": 2, "phase": "reduce", "via": "HierarchicalReducer"},
        rowid=7,
    )
    d = e.to_dict()
    assert d["provenance"]["phase"] == "reduce"
    assert "rowid" not in d

    e2 = Edge.from_dict(d)
    assert e2.provenance == e.provenance
    assert e2.rowid is None
