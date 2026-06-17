"""Test ontology constants and dataclasses (v0.5: 5 levels, decoupled type)."""

import pytest
from ccchain.core.ontology import (
    ATOM_TYPES,
    ATOM_TYPE_SET,
    CC_EDGE_TYPES,
    HIERARCHY_EDGES,
    LEVELS,
    LEVEL_ORDER,
    LEVEL_ALIAS,
    LEVEL_DEFAULT_TYPE,
    STRONG_CAUSAL_EDGES,
    TYPE_COMPATIBILITY,
    TYPE_TO_COE_CHECKS,
    Atom,
    Edge,
    Rho,
    TaskSpec,
    Trajectory,
)


def test_atom_types_count():
    assert len(ATOM_TYPES) == 12


def test_levels_order():
    assert len(LEVELS) == 5
    assert LEVEL_ORDER["W1_problem"] == 0
    assert LEVEL_ORDER["W2_direction"] == 1
    assert LEVEL_ORDER["W3_approach"] == 2
    assert LEVEL_ORDER["W4_implementation"] == 3
    assert LEVEL_ORDER["W5_code"] == 4


def test_level_alias_mapping():
    assert LEVEL_ALIAS["W1"] == "W1_problem"
    assert LEVEL_ALIAS["W2"] == "W2_direction"
    assert LEVEL_ALIAS["W3"] == "W3_approach"
    assert LEVEL_ALIAS["W4"] == "W4_implementation"
    assert LEVEL_ALIAS["W5"] == "W5_code"


def test_level_default_type_has_5_entries():
    assert len(LEVEL_DEFAULT_TYPE) == 5
    for lvl in LEVELS:
        assert LEVEL_DEFAULT_TYPE[lvl] in ATOM_TYPE_SET


def test_strong_causal_edges():
    assert "extends" in STRONG_CAUSAL_EDGES
    assert "improves" in STRONG_CAUSAL_EDGES
    assert "related_to" not in STRONG_CAUSAL_EDGES


def test_hierarchy_edges():
    assert "aggregates_to" in HIERARCHY_EDGES
    assert "decomposes_into" in HIERARCHY_EDGES


def test_type_to_coe_checks_mapping():
    """CoE check mapping per type (v0.5: keyed by type, level-independent)."""
    assert TYPE_TO_COE_CHECKS["numerical"] == {"I1"}
    assert TYPE_TO_COE_CHECKS["citation"] == {"I3"}
    assert TYPE_TO_COE_CHECKS["method"] == {"I4"}
    assert TYPE_TO_COE_CHECKS["solution"] == {"I4"}
    assert TYPE_TO_COE_CHECKS["experiment"] == {"I2"}
    assert TYPE_TO_COE_CHECKS.get("problem", set()) == set()
    assert TYPE_TO_COE_CHECKS.get("concept", set()) == set()
    assert TYPE_TO_COE_CHECKS.get("conclusion", set()) == set()
    assert TYPE_TO_COE_CHECKS.get("component", set()) == set()
    assert TYPE_TO_COE_CHECKS.get("verification", set()) == set()


def test_type_compatibility():
    assert ("method", "method") in TYPE_COMPATIBILITY["extends"]
    assert ("solution", "solution") in TYPE_COMPATIBILITY["extends"]
    assert ("component", "component") in TYPE_COMPATIBILITY["extends"]
    assert ("method", "component") in TYPE_COMPATIBILITY["uses_component"]
    assert ("citation", "method") in TYPE_COMPATIBILITY["background"]


def test_atom_creation():
    a = Atom(
        node_id="W4_test_1", name="Test Solution", type="solution",
        level="W4_implementation", context="Test context",
    )
    assert a.node_id == "W4_test_1"
    assert a.type == "solution"
    assert a.status == "active"


def test_atom_invalid_type():
    with pytest.raises(ValueError):
        Atom(node_id="X", name="Bad", type="invalid_type", level="W4_implementation")


def test_atom_invalid_level():
    with pytest.raises(ValueError):
        Atom(node_id="X", name="Bad", type="solution", level="W99")


def test_atom_type_level_decoupled():
    """v0.5: type and level are decoupled — ANY type constructs at ANY level."""
    for t in ATOM_TYPES:
        for lvl in LEVELS:
            a = Atom(node_id=f"x_{t}_{lvl}", name="n", type=t, level=lvl)
            assert a.type == t and a.level == lvl


def test_atom_to_from_dict():
    a = Atom(
        node_id="W4_test", name="Test", type="method",
        level="W2_direction", context="Ctx", tags=["domain:MARL"],
    )
    d = a.to_dict()
    assert d["node_id"] == "W4_test"
    a2 = Atom.from_dict(d)
    assert a2.name == "Test"
    assert a2.tags == ["domain:MARL"]


def test_edge_creation():
    rho = Rho(bottleneck="credit_assignment", mechanism="OT distance",
              tradeoff="Computational cost", confidence=0.9)
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
    for lvl in ["W1", "W2", "W3", "W4", "W5"]:
        assert embs[lvl].shape == (0, 1024)


def test_atom_v02_fields_default_none():
    a = Atom(node_id="W5_test", name="Sinkhorn", type="component", level="W5_code")
    assert a.code_body is None
    assert a.source_refs is None
    assert a.provenance is None
    assert a.rowid is None


def test_atom_v02_fields_roundtrip():
    a = Atom(
        node_id="W5_test", name="Sinkhorn", type="component", level="W5_code",
        code_body="def sinkhorn2(cost, reg=0.1): ...",
        source_refs=["arxiv:2203.12345", "paper:OT-CTDE-2024"],
        provenance={"cycle": 1, "phase": "extract", "via": "TwoPhaseExtractor"},
        rowid=42,
    )
    d = a.to_dict()
    assert d["code_body"] == "def sinkhorn2(cost, reg=0.1): ..."
    assert d["source_refs"] == ["arxiv:2203.12345", "paper:OT-CTDE-2024"]
    assert d["provenance"]["cycle"] == 1
    assert "rowid" not in d
    a2 = Atom.from_dict(d)
    assert a2.code_body == a.code_body
    assert a2.source_refs == a.source_refs
    assert a2.provenance == a.provenance
    assert a2.rowid is None


def test_edge_v02_provenance_roundtrip():
    e = Edge(src="A", relation="extends", tgt="B",
             provenance={"cycle": 2, "phase": "reduce", "via": "HierarchicalReducer"}, rowid=7)
    d = e.to_dict()
    assert d["provenance"]["phase"] == "reduce"
    assert "rowid" not in d
    e2 = Edge.from_dict(d)
    assert e2.provenance == e.provenance
    assert e2.rowid is None


def test_taskspec_creation():
    ts = TaskSpec(task_name="smac-1v1", eval_harness="smac-v1",
                  success_criteria="win_rate > 0.9 against built-in AI",
                  constraints=["CTDE", "no centralised critic"])
    assert ts.task_name == "smac-1v1"
    assert len(ts.constraints) == 2


def test_taskspec_roundtrip():
    ts = TaskSpec(task_name="montezuma", eval_harness="ale", success_criteria="score > 6000")
    d = ts.to_dict()
    assert d["task_name"] == "montezuma"
    ts2 = TaskSpec.from_dict(d)
    assert ts2.eval_harness == "ale"
    assert ts2.constraints == []
