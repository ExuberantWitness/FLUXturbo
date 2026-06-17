"""Test CCStore v0.4 → v0.5 level-string migration + decoupled dual-write."""

import os
import tempfile

import pytest

from ccchain.core.ontology import (
    ATOM_TYPES,
    LEVEL_DEFAULT_TYPE,
    LEVEL_MIGRATION_V04_TO_V05,
    LEVELS,
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
    """Insert atom directly into SQLite (bypassing Atom.__post_init__).

    Used to simulate legacy data that needs migration.
    """
    now = "2025-01-01T00:00:00"
    store.db.execute(
        """INSERT INTO cc_nodes
           (node_id, name, type, level, context, version, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (node_id, name, atom_type, level, kwargs.get("context", ""),
         1, "active", now, now),
    )
    store.db.commit()


def test_12_type_dual_write_decoupled(temp_store):
    """All 12 types insert + query. v0.5: type and level are decoupled, so each
    type is placed at a level via LEVEL_DEFAULT_TYPE (any type fits any level)."""
    atoms = []
    for i, t in enumerate(ATOM_TYPES):
        lvl = LEVELS[i % len(LEVELS)]
        a = Atom(node_id=f"test_{t}", name=f"Test {t}", type=t, level=level_ok(lvl),
                 context=f"Test {t} atom")
        atoms.append(a)

    result = temp_store.insert_blueprint(atoms, [], "test.pdf")
    assert result["inserted_nodes"] == 12

    # Each atom queryable at its (decoupled) level
    for i, t in enumerate(ATOM_TYPES):
        lvl = LEVELS[i % len(LEVELS)]
        lvl_atoms = temp_store.query_by_level(level_ok(lvl))
        assert f"test_{t}" in [a.node_id for a in lvl_atoms]


def level_ok(lvl):
    return lvl


# ---------------------------------------------------------------------------
# v0.4 → v0.5 level-string migration
# ---------------------------------------------------------------------------
def test_migration_level_strings(temp_store):
    """Old 4-layer level strings rewrite to new 5-layer strings on reopen."""
    for old, new in LEVEL_MIGRATION_V04_TO_V05.items():
        _insert_raw(temp_store, f"old_{old}", f"Old {old}", "concept", old)

    db_path, graph_dir = temp_store.db_path, temp_store.graph_dir
    temp_store.db.close()
    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)

    for old, new in LEVEL_MIGRATION_V04_TO_V05.items():
        atom = store2.query_by_id(f"old_{old}")
        assert atom is not None
        assert atom.level == new, f"{old} should migrate to {new}, got {atom.level}"


def test_migration_paper_to_citation(temp_store):
    """Legacy 'paper' type renames to 'citation' (level unchanged)."""
    _insert_raw(temp_store, "old_paper_1", "Old Paper", "paper", "W2_direction")
    db_path, graph_dir = temp_store.db_path, temp_store.graph_dir
    temp_store.db.close()
    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("old_paper_1")
    assert atom.type == "citation"
    assert atom.level == "W2_direction"  # level string already new-style → unchanged


def test_migration_fact_to_concept(temp_store):
    """Legacy 'fact' type renames to 'concept'."""
    _insert_raw(temp_store, "old_fact_1", "Old Fact", "fact", "W2_direction")
    db_path, graph_dir = temp_store.db_path, temp_store.graph_dir
    temp_store.db.close()
    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("old_fact_1")
    assert atom.type == "concept"
    assert atom.level == "W2_direction"


def test_migration_idempotent(temp_store):
    """Migration is idempotent — reopening twice is safe."""
    _insert_raw(temp_store, "old_paper_1", "Old Paper", "paper",
                "W2_problem_analysis")  # old level string
    db_path, graph_dir = temp_store.db_path, temp_store.graph_dir
    temp_store.db.close()

    store1 = CCStore(db_path=db_path, graph_dir=graph_dir)
    a1 = store1.query_by_id("old_paper_1")
    assert a1.type == "citation" and a1.level == "W1_problem"
    store1.db.close()

    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    a2 = store2.query_by_id("old_paper_1")
    assert a2.type == "citation" and a2.level == "W1_problem"


def test_decoupled_type_not_forced_to_level(temp_store):
    """v0.5: a 'solution' at W5_code is NOT rewritten to W4 — types are decoupled."""
    _insert_raw(temp_store, "weird_1", "Weird", "solution", "W5_code")
    db_path, graph_dir = temp_store.db_path, temp_store.graph_dir
    temp_store.db.close()
    store2 = CCStore(db_path=db_path, graph_dir=graph_dir)
    atom = store2.query_by_id("weird_1")
    assert atom.type == "solution"
    assert atom.level == "W5_code"  # decoupled — no canonical-level forcing
