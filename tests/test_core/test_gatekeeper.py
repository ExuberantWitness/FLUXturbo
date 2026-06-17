"""Test 7-layer gatekeeper validation (v0.3)."""

import pytest

from ccchain.core.gatekeeper import (
    PROVENANCE_REQUIREMENTS,
    TYPE_DEMOTION_MAP,
    apply_r6_demotions,
    validate,
)
from ccchain.core.ontology import Atom, Edge, Rho


def _atom(node_id, name, type, level, context="test", provenance=None):
    """Build an atom that already passes Atom.__post_init__ (valid type+level)."""
    return Atom(
        node_id=node_id, name=name, type=type, level=level,
        context=context, provenance=provenance,
    )


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


# ---------------------------------------------------------------------------
# R1 — Schema
# ---------------------------------------------------------------------------
def test_validate_empty():
    assert validate([], []) == []


def test_validate_valid_atoms():
    atoms = [
        _atom("w2_1", "Problem X", "bottleneck", "W1_problem"),
        _atom("w3_1", "Solution Y", "method", "W2_direction",
              provenance={"code_span": "lines 5-20"}),
    ]
    assert validate(atoms, []) == []


def test_r1_atom_invalid_type():
    atoms = [_make_atom("x", "Bad", "not_a_type", "W4_implementation")]
    errors = validate(atoms, [])
    assert any(e["rule"] == "R1" for e in errors)


def test_r1_atom_invalid_level():
    atoms = [_make_atom("x", "Bad", "solution", "W99")]
    errors = validate(atoms, [])
    assert any(e["rule"] == "R1" for e in errors)


def test_r1_invalid_edge_relation():
    atoms = [
        _atom("a", "A", "method", "W2_direction"),
        _atom("b", "B", "method", "W2_direction"),
    ]
    edges = [_make_edge("a", "not_a_relation", "b")]
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R1" and "relation" in e["message"].lower() for e in errors)


def test_atom_constructor_rejects_invalid_type():
    with pytest.raises(ValueError, match="Invalid atom type"):
        _atom("x", "Bad", "not_a_type", "W4_implementation")


def test_atom_constructor_rejects_invalid_level():
    with pytest.raises(ValueError, match="Invalid level"):
        _atom("x", "Bad", "solution", "W99")


def test_edge_constructor_rejects_invalid_relation():
    with pytest.raises(ValueError, match="Invalid edge relation"):
        Edge(src="a", relation="not_a_relation", tgt="b")


def test_r1_edge_endpoint_missing():
    atoms = [_atom("a", "A", "method", "W2_direction")]
    edges = [_make_edge("a", "extends", "nonexistent")]
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R1" for e in errors)


# ---------------------------------------------------------------------------
# R2 — Type Compatibility
# ---------------------------------------------------------------------------
def test_r2_type_incompatibility():
    atoms = [
        _atom("a", "A", "method", "W2_direction"),
        _atom("b", "B", "citation", "W2_direction"),
    ]
    # extends requires method→method or solution→solution or component→component
    edges = [_make_edge("a", "extends", "b")]
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R2" for e in errors)


# ---------------------------------------------------------------------------
# R3 — Rho Completeness
# ---------------------------------------------------------------------------
def test_r3_missing_rho():
    atoms = [
        _atom("a", "A", "method", "W2_direction"),
        _atom("b", "B", "method", "W2_direction"),
    ]
    edges = [_make_edge("a", "extends", "b")]  # no rho
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R3" for e in errors)


def test_r3_with_rho_passes():
    atoms = [
        _atom("a", "A", "method", "W2_direction"),
        _atom("b", "B", "method", "W2_direction"),
    ]
    rho = Rho(bottleneck="test", mechanism="m", tradeoff="t", confidence=0.8)
    edges = [Edge(src="a", relation="extends", tgt="b", rho=rho)]
    errors = validate(atoms, edges)
    r3_errors = [e for e in errors if e["rule"] == "R3"]
    assert len(r3_errors) == 0


# ---------------------------------------------------------------------------
# R4 — Level Consistency (hierarchy edges)
# ---------------------------------------------------------------------------
def test_r4_level_direction_wrong():
    atoms = [
        _atom("a", "W4", "solution", "W4_implementation"),
        _atom("b", "W5", "component", "W5_code"),
    ]
    # aggregates_to must go upward (W5→W4), but here src=W4 tgt=W5
    edges = [_make_edge("a", "aggregates_to", "b")]
    errors = validate(atoms, edges)
    assert any(e["rule"] == "R4" for e in errors)


def test_r4_level_direction_correct():
    atoms = [
        _atom("a", "W5_impl", "component", "W5_code"),
        _atom("b", "W4_sol", "solution", "W4_implementation"),
    ]
    edges = [_make_edge("a", "aggregates_to", "b")]
    errors = validate(atoms, edges)
    r4 = [e for e in errors if e["rule"] == "R4"]
    assert len(r4) == 0


# ---------------------------------------------------------------------------
# R5 — Dedup Detection
# ---------------------------------------------------------------------------
def test_r5_dedup_detection():
    atoms = [
        _atom("a", "Same Name", "method", "W2_direction"),
        _atom("b", "Same Name", "method", "W2_direction"),
    ]
    errors = validate(atoms, [])
    assert any(e["rule"] == "R5" for e in errors)


def test_r5_no_dedup_different_type():
    atoms = [
        _atom("a", "Same Name", "method", "W2_direction"),
        _atom("b", "Same Name", "citation", "W2_direction"),
    ]
    errors = validate(atoms, [])
    assert not any(e["rule"] == "R5" for e in errors)


# ---------------------------------------------------------------------------
# R6 — Provenance Presence (by type)
# ---------------------------------------------------------------------------
def test_r6_numerical_missing_score():
    """numerical atom requires provenance.score."""
    a = _atom("n1", "Score", "numerical", "W4_implementation",
              provenance={"raw_citation": "x"})  # score key missing
    errors = validate([a], [])
    assert any(e["rule"] == "R6" for e in errors)


def test_r6_numerical_with_score_passes():
    a = _atom("n1", "Score", "numerical", "W4_implementation",
              provenance={"score": 0.95, "score_std": 0.01})
    errors = validate([a], [])
    r6 = [e for e in errors if e["rule"] == "R6"]
    assert len(r6) == 0


def test_r6_citation_missing_raw_citation():
    a = _atom("c1", "Vaswani et al", "citation", "W2_direction",
              provenance={"year": 2017})  # raw_citation missing
    errors = validate([a], [])
    assert any(e["rule"] == "R6" for e in errors)


def test_r6_citation_with_raw_citation_passes():
    a = _atom("c1", "Vaswani et al", "citation", "W2_direction",
              provenance={"raw_citation": "Vaswani et al. 2017 Attention is all you need"})
    errors = validate([a], [])
    r6 = [e for e in errors if e["rule"] == "R6"]
    assert len(r6) == 0


def test_r6_method_requires_any_provenance():
    a = _atom("m1", "Transformer", "method", "W2_direction",
              provenance=None)
    errors = validate([a], [])
    assert any(e["rule"] == "R6" for e in errors)


def test_r6_subjective_types_no_provenance_required():
    """problem/bottleneck/hypothesis/concept/conclusion don't need provenance."""
    atoms = [
        _atom("p1", "P", "problem", "W1_problem"),
        _atom("b1", "B", "bottleneck", "W1_problem"),
        _atom("h1", "H", "hypothesis", "W1_problem"),
        _atom("c1", "C", "concept", "W2_direction"),
        _atom("cn1", "CN", "conclusion", "W4_implementation"),
    ]
    errors = validate(atoms, [])
    r6 = [e for e in errors if e["rule"] == "R6"]
    assert len(r6) == 0


# ---------------------------------------------------------------------------
# R7 — Type-Level Consistency (REMOVED in v0.5; type and level are decoupled)
# ---------------------------------------------------------------------------
def test_r7_removed_type_level_decoupled():
    """v0.5: any type at any level passes validation (no R7 errors)."""
    a = _make_atom("x", "Any", "numerical", "W5_code")
    errors = validate([a], [])
    assert not any(e["rule"] == "R7" for e in errors)


def test_atom_constructor_decoupled():
    """v0.5: Atom constructor accepts any (type, level) pair without error."""
    a = Atom(node_id="x", name="Any", type="numerical", level="W5_code")
    assert a.type == "numerical" and a.level == "W5_code"


# ---------------------------------------------------------------------------
# apply_r6_demotions — fallback demotion logic (v0.5: type changes, level stays)
# ---------------------------------------------------------------------------
def test_apply_r6_demotions_numerical_to_conclusion():
    """numerical atom missing score → demoted to conclusion (level unchanged)."""
    a = _atom("n1", "Score", "numerical", "W4_implementation", provenance=None)
    n = apply_r6_demotions([a])
    assert n == 1
    assert a.type == "conclusion"
    assert a.level == "W4_implementation"  # decoupled — level unchanged
    assert a.status == "demoted"


def test_apply_r6_demotions_citation_to_concept():
    """citation atom missing raw_citation → demoted to concept."""
    a = _atom("c1", "Vaswani", "citation", "W2_direction", provenance=None)
    n = apply_r6_demotions([a])
    assert n == 1
    assert a.type == "concept"
    assert a.level == "W2_direction"
    assert a.status == "demoted"


def test_apply_r6_demotions_method_marks_needs_review():
    """method atom without provenance → no demotion target, marked needs_review."""
    a = _atom("m1", "Transformer", "method", "W2_direction", provenance=None)
    n = apply_r6_demotions([a])
    assert n == 1
    assert a.type == "method"  # unchanged
    assert a.status == "needs_review"


def test_apply_r6_demotions_idempotent_on_valid():
    """Atoms already satisfying R6 are not mutated."""
    a = _atom("n1", "Score", "numerical", "W4_implementation",
              provenance={"score": 0.9})
    n = apply_r6_demotions([a])
    assert n == 0
    assert a.type == "numerical"
    assert a.status == "active"


def test_apply_r6_demotions_no_op_for_subjective_types():
    """Subjective types are skipped."""
    atoms = [
        _atom("p1", "P", "problem", "W1_problem"),
        _atom("c1", "C", "concept", "W2_direction"),
    ]
    n = apply_r6_demotions(atoms)
    assert n == 0


def test_provenance_requirements_coverage():
    """All CoE-triggering types have a requirements entry."""
    assert set(PROVENANCE_REQUIREMENTS.keys()) == {
        "numerical", "citation", "method", "solution", "experiment"
    }


def test_demotion_map_types_valid():
    """v0.5: demotion targets must be valid atom types (level is decoupled, so
    there's no same-layer constraint — any valid type can be a demotion target)."""
    from ccchain.core.ontology import ATOM_TYPE_SET
    for old_t, new_t in TYPE_DEMOTION_MAP.items():
        assert old_t in ATOM_TYPE_SET and new_t in ATOM_TYPE_SET, \
            f"{old_t}→{new_t} uses invalid types"
