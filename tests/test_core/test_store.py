"""Test CCStore v0.2 → v0.3 type migration and 12-type dual-write."""

import os
import tempfile

import pytest

from ccchain.core.ontology import (
    TYPE_TO_LEVEL,
    Atom,
    Edge,
)
from ccchain.core.store import CCStore


@pytest.fixture
def temp_store():
    """Fresh CCStore in a temp dir for each test."""
    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_obj.name
    db_path = os.path.join(tmpdir, "test.db")
    graph_dir = os.path.join(tmpdir, "graph")
    store = CCStore(db_path=db_path, graph_dir=graph_dir)
    yield store
    try:
        store.db.close()
    except Exception:
        pass
    try:
        tmpdir_obj.cleanup()
    except Exception:
        pass


def _insert_raw(store, node_id, name, atom_type, level, **kwargs):
    """Insert atom directly into SQLite (bypassing Atom.__post_init__ validation).

    Used to simulate v0.2 data that needs migration.
    """
    import json
    now = "2025-01-01T00:00:00"
    store.db.execute(
        """INSERT INTO cc_nodes
           (node_id, name, type, level, context, version, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (node_id, name, atom_type, level, kwargs.get("context", ""),
         1, "active", now, now),
    )
    store.db.commit()


def test_12_type_dual_write(temp_store):
    """All 12 v0.3 types can be inserted and queried."""
    atoms = []
    for i, (t, lvl) in enumerate(TYPE_TO_LEVEL.items()):
        a = Atom(
            node_id=f"test_{t}",
            name=f"Test {t}",
            type=t,
            level=lvl,
            context=f"Test {t} atom",
        )
        a.provenance = {"stub": True} if t in ("numerical", "citation", "method", "solution", "experiment") else None
        if t == "numerical":
            a.provenance = {"score": 0.5}
        elif t == "citation":
            a.provenance = {"raw_citation": "test"}
        atoms.append(a)

    result = temp_store.insert_blueprint(atoms, [], "test.pdf")
    assert result["inserted_nodes"] == 12

    # All atoms should be queryable by their canonical level
    for t, lvl in TYPE_TO_LEVEL.items():
        lvl_atoms = temp_store.query_by_level(lvl)
        ids = [a.node_id for a in lvl_atoms]
        assert f"test_{t}" in ids


def test_migration_paper_to_citation(temp_store):
    """v0.2 'paper' atom migrates to v0.3 'citation' at W3."""
    _insert_raw(temp_store, "old_paper_1", "Old Paper", "paper", "W3_solution_direction")

    # Re-instantiate store — migration runs in __init__
    db_path = temp_store.db_path
    graph_dir = temp_store.graph_dir
    temp_store.db.close()

    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("old_paper_1")
    assert atom is not None
    assert atom.type == "citation"
    assert atom.level == "W3_solution_direction"


def test_migration_fact_to_concept(temp_store):
    """v0.2 'fact' atom migrates to v0.3 'concept' at W3."""
    _insert_raw(temp_store, "old_fact_1", "Old Fact", "fact", "W3_solution_direction")

    db_path = temp_store.db_path
    graph_dir = temp_store.graph_dir
    temp_store.db.close()

    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("old_fact_1")
    assert atom is not None
    assert atom.type == "concept"
    assert atom.level == "W3_solution_direction"


def test_migration_w4_method_to_solution(temp_store):
    """v0.2 W4 'method' atom migrates to v0.3 W4 'solution'."""
    _insert_raw(temp_store, "old_w4_method", "Old W4 Method", "method", "W4_concrete_solution")

    db_path = temp_store.db_path
    graph_dir = temp_store.graph_dir
    temp_store.db.close()

    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("old_w4_method")
    assert atom is not None
    assert atom.type == "solution"
    assert atom.level == "W4_concrete_solution"


def test_migration_w3_method_stays_method(temp_store):
    """v0.2 W3 'method' atom stays 'method' at W3."""
    _insert_raw(temp_store, "old_w3_method", "Old W3 Method", "method", "W3_solution_direction")

    db_path = temp_store.db_path
    graph_dir = temp_store.graph_dir
    temp_store.db.close()

    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("old_w3_method")
    assert atom is not None
    assert atom.type == "method"
    assert atom.level == "W3_solution_direction"


def test_migration_idempotent(temp_store):
    """Migration is idempotent — running twice is safe."""
    _insert_raw(temp_store, "old_paper_1", "Old Paper", "paper", "W3_solution_direction")

    db_path = temp_store.db_path
    graph_dir = temp_store.graph_dir
    temp_store.db.close()

    # First migration
    store1 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom1 = store1.query_by_id("old_paper_1")
    assert atom1.type == "citation"
    store1.db.close()

    # Second migration (should be no-op for already-migrated atoms)
    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom2 = store2.query_by_id("old_paper_1")
    assert atom2.type == "citation"


def test_migration_enforces_canonical_level(temp_store):
    """Even if an atom has the right type but wrong level, migration fixes it."""
    # Insert a v0.3-style atom but with wrong level (simulating manual edit / corruption)
    _insert_raw(temp_store, "weird_1", "Weird", "solution", "W5_code_implementation")

    db_path = temp_store.db_path
    graph_dir = temp_store.graph_dir
    temp_store.db.close()

    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("weird_1")
    assert atom.type == "solution"
    assert atom.level == "W4_concrete_solution"  # canonical level enforced
