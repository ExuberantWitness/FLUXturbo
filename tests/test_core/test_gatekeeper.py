"""Test 5-layer gatekeeper validation."""

from dataclasses import dataclass

import pytest

from ccchain.core.gatekeeper import validate
from ccchain.core.ontology import Atom, Edge, Rho


def _atom(node_id, name, type, level, context="test"):
    return Atom(node_id=node_id, name=name, type=type, level=level, context=context)


def _make_atom(node_id, name, type, level, context="test"):
    """Create Atom bypassing __post_init__ validation (simulates external data)."""
    a = Atom.__new__(Atom)
    a.node_id = node_id
    a.name = name
    a.type = type
    a.level = level
    a.context = context
    a.version = 1
    a.source_pdf = None
    a.source_chunk = None
    a.code_ref = None
    a.references = None
    a.tags = None
    a.status = "active"
    a.embedding = None
    a.created_at = None
    a.updated_at = None
    a.code_body = None
    a.source_refs = None
    a.provenance = None
    a.rowid = None
    return a


def _make_edge(src, relation, tgt, weight=1.0, rho=None):
    """Create Edge bypassing __post_init__ validation."""
    e = Edge.__new__(Edge)
    e.src = src
    e.relation = relation
    e.tgt = tgt
    e.weight = weight
    e.rho = rho
    e.provenance = None
    e.rowid = None
    return e


def test_validate_empty():
    assert validate([], []) == []


def test_validate_valid_atoms():
    atoms = [
        _atom("w2_1", "Problem X", "bottleneck", "W2_problem_analysis"),
        _atom("w3_1", "Solution Y", "method", "W3_solution_direction"),
    ]
    assert validate(atoms, []) == []


def test_r1_atom_invalid_type():
    """Gatekeeper catches invalid atom type (from deserialized data)."""
    atoms = [_make_atom("x", "Bad", "not_a_type", "W4_concrete_solution")]
    errors = validate(atoms, [])
    assert any(e["rule"] == "R1" for e in errors)


def test_r1_atom_invalid_level():
    """Gatekeeper catches invalid atom level (from deserialized data)."""
    atoms = [_make_atom("x", "Bad", "method", "W99")]
    errors = validate(atoms, [])
    assert any(e["rule"] == "R1" for e in errors)


def test_r1_invalid_edge_relation():
    """Gatekeeper catches invalid edge relation (from deserialized data)."""
    atoms = [
        _atom("a", "A", "method", "W4_concrete_solution"),
        _atom("b", "B", "method", "W4_concrete_solution"),
    ]
    edges = [_make_edge("a", "not_a_relation", "b")]
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R1" and "relation" in e["message"].lower() for e in errors)


def test_atom_constructor_rejects_invalid_type():
    with pytest.raises(ValueError, match="Invalid atom type"):
        _atom("x", "Bad", "not_a_type", "W4_concrete_solution")


def test_atom_constructor_rejects_invalid_level():
    with pytest.raises(ValueError, match="Invalid level"):
        _atom("x", "Bad", "method", "W99")


def test_edge_constructor_rejects_invalid_relation():
    with pytest.raises(ValueError, match="Invalid edge relation"):
        Edge(src="a", relation="not_a_relation", tgt="b")


def test_r1_edge_endpoint_missing():
    atoms = [_atom("a", "A", "method", "W4_concrete_solution")]
    edges = [_make_edge("a", "extends", "nonexistent")]
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R1" for e in errors)


def test_r2_type_incompatibility():
    atoms = [
        _atom("a", "A", "method", "W4_concrete_solution"),
        _atom("b", "B", "paper", "W4_concrete_solution"),
    ]
    edges = [_make_edge("a", "extends", "b")]  # extends only allows method→method
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R2" for e in errors)


def test_r3_missing_rho():
    atoms = [
        _atom("a", "A", "method", "W4_concrete_solution"),
        _atom("b", "B", "method", "W4_concrete_solution"),
    ]
    edges = [_make_edge("a", "extends", "b")]  # no rho
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R3" for e in errors)


def test_r3_with_rho_passes():
    atoms = [
        _atom("a", "A", "method", "W4_concrete_solution"),
        _atom("b", "B", "method", "W4_concrete_solution"),
    ]
    rho = Rho(bottleneck="test", mechanism="m", tradeoff="t", confidence=0.8)
    edges = [Edge(src="a", relation="extends", tgt="b", rho=rho)]
    errors = validate(atoms, edges)
    r3_errors = [e for e in errors if e["rule"] == "R3"]
    assert len(r3_errors) == 0


def test_r4_level_direction_wrong():
    atoms = [
        _atom("a", "W4", "method", "W4_concrete_solution"),
        _atom("b", "W5", "component", "W5_code_implementation"),
    ]
    edges = [_make_edge("a", "aggregates_to", "b")]
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R4" for e in errors)


def test_r4_level_direction_correct():
    atoms = [
        _atom("a", "W5_impl", "component", "W5_code_implementation"),
        _atom("b", "W4_sol", "method", "W4_concrete_solution"),
    ]
    edges = [_make_edge("a", "aggregates_to", "b")]
    errors = validate(atoms, edges)
    r4 = [e for e in errors if e["rule"] == "R4"]
    assert len(r4) == 0


def test_r5_dedup_detection():
    atoms = [
        _atom("a", "Same Name", "method", "W4_concrete_solution"),
        _atom("b", "Same Name", "method", "W4_concrete_solution"),
    ]
    errors = validate(atoms, [])
    assert any(e["rule"] == "R5" for e in errors)


def test_r5_no_dedup_different_type():
    atoms = [
        _atom("a", "Same Name", "method", "W4_concrete_solution"),
        _atom("b", "Same Name", "bottleneck", "W4_concrete_solution"),
    ]
    errors = validate(atoms, [])
    assert not any(e["rule"] == "R5" for e in errors)
