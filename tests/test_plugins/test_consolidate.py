"""Test cross-paper hierarchical consolidation (ccchain.consolidate)."""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from ccchain.config import Config
from ccchain.consolidate import (
    HierarchicalConsolidator,
    _candidate_clusters,
    _cosine_matrix,
    consolidate,
)
from ccchain.core.ontology import Atom, Edge
from ccchain.core.store import CCStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store():
    tmp = tempfile.mkdtemp(prefix="ccchain_consol_")
    s = CCStore(db_path=os.path.join(tmp, "t.db"), graph_dir=tmp)
    yield s
    s.db.close()


def _embed_factory():
    """Deterministic embed keyed by an injected tag in the text.

    Tests embed atoms whose ``context`` carries ``[emb:KEY]``; atoms with the
    same KEY get near-identical vectors (cosine ~1), different KEY → orthogonal.
    """
    def _embed(texts, **kwargs):
        base = {
            "OT": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "SHA": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "CTDE": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        }
        out = []
        for t in texts:
            key = "DEFAULT"
            for k in base:
                if f"[emb:{k}]" in t:
                    key = k
                    break
            v = base.get(key, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)).copy()
            out.append(v + np.float32(1e-3))  # tiny jitter, still cosine ~1 within key
        return np.stack(out).astype(np.float32)
    return _embed


def _arbiter_factory(merge: bool, canonical_name: str = "Unified OT Method"):
    """LLM arbiter mock returning a fixed verdict."""
    def _fn(messages, **kwargs):
        return {
            "merge": merge,
            "canonical_name": canonical_name,
            "canonical_type": "method",
            "canonical_context": "Unified optimal-transport credit assignment.",
            "reasoning": "mock arbiter",
        }
    return _fn


def _two_papers_ot_under_same_parent(store):
    """Insert two W3 'OT method' atoms under a shared W2 parent, both with OT
    embeddings (so cosine ~1). Returns (w2_id, w3_a_id, w3_b_id)."""
    w2 = Atom(node_id="w2_root", name="RL Problem", type="bottleneck",
              level="W1_problem", context="credit assignment noise",
              provenance={})
    w3a = Atom(node_id="w3_a", name="OT Method A", type="method",
               level="W2_direction", context="OT credit [emb:OT]",
               provenance={"code_span": "s"})
    w3b = Atom(node_id="w3_b", name="OT Method B", type="method",
               level="W2_direction", context="optimal transport credit [emb:OT]",
               provenance={"code_span": "s"})
    edges = [
        Edge(src="w2_root", relation="decomposes_into", tgt="w3_a"),
        Edge(src="w2_root", relation="decomposes_into", tgt="w3_b"),
        Edge(src="w3_a", relation="aggregates_to", tgt="w2_root"),
        Edge(src="w3_b", relation="aggregates_to", tgt="w2_root"),
    ]
    # Embed both W3 atoms so they land in the embedding table.
    from ccchain.core.embedding import embed as _real_embed  # patched below normally
    for a in (w3a, w3b):
        a.embedding = _embed_factory()([a.context])[0]
    store.insert_blueprint([w2, w3a, w3b], edges, "p.pdf")
    return "w2_root", "w3_a", "w3_b"


# ---------------------------------------------------------------------------
# Unit tests (no LLM, no store)
# ---------------------------------------------------------------------------
def test_cosine_matrix_basic():
    v = [np.array([1, 0, 0], dtype=np.float32),
         np.array([1, 0, 0], dtype=np.float32),
         np.array([0, 1, 0], dtype=np.float32)]
    M = _cosine_matrix(v)
    assert abs(float(M[0, 1]) - 1.0) < 1e-4
    assert abs(float(M[0, 2])) < 1e-4


def test_candidate_clusters_groups_similar():
    v = {f"n{i}": vec for i, vec in enumerate([
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0.99, 0.01, 0], dtype=np.float32),  # ~1 with n0
        np.array([0, 1, 0], dtype=np.float32),         # distinct
    ])}
    atoms = [Atom(node_id=k, name=k, type="method", level="W2_direction")
             for k in v]
    clusters = _candidate_clusters(atoms, v, 0.85)
    assert len(clusters) == 1
    assert {a.node_id for a in clusters[0]} == {"n0", "n1"}


def test_candidate_clusters_singletons_dropped():
    v = {"n0": np.array([1, 0], dtype=np.float32),
         "n1": np.array([0, 1], dtype=np.float32)}
    atoms = [Atom(node_id=k, name=k, type="method", level="W2_direction")
             for k in v]
    assert _candidate_clusters(atoms, v, 0.85) == []


# ---------------------------------------------------------------------------
# Integration scenarios (mocked LLM + embed)
# ---------------------------------------------------------------------------
def _cfg(store):
    return Config(
        db_path=store.db_path_unused if False else "ignored",
        consolidate_similarity_threshold=0.85, consolidate_majority_k=1,
        reference_api_timeout=0.1, reference_api_max_retries=1,
    )


@patch("ccchain.core.embedding.embed")
@patch("ccchain.core.llm.chat_json_majority")
def test_consolidate_merges_duplicates(mock_majority, mock_embed, store):
    """(a) two same-concept W3 atoms under same W2 parent → 1 canonical, 1 merged."""
    mock_embed.side_effect = _embed_factory()
    mock_majority.side_effect = _arbiter_factory(merge=True)
    _two_papers_ot_under_same_parent(store)

    report = consolidate(store, config=Config(
        consolidate_similarity_threshold=0.85, consolidate_majority_k=1))

    assert report["atoms_merged"] == 1
    # canonical is cluster[0] = w3_a (insertion order), w3_b merged away
    a = store.query_by_id("w3_a")
    b = store.query_by_id("w3_b")
    assert a.status == "active"
    assert a.name == "Unified OT Method"          # LLM-rewritten
    assert b.status == "merged"
    assert b.provenance["merged_into"] == "w3_a"
    # only one live W3 remains
    live_w3 = [x for x in store.query_by_level("W2_direction", status=None)
               if x.status != "merged"]
    assert len(live_w3) == 1


@patch("ccchain.core.embedding.embed")
@patch("ccchain.core.llm.chat_json_majority")
def test_consolidate_idempotent(mock_majority, mock_embed, store):
    """(b) running twice → no new merges on second pass."""
    mock_embed.side_effect = _embed_factory()
    mock_majority.side_effect = _arbiter_factory(merge=True)
    _two_papers_ot_under_same_parent(store)

    r1 = consolidate(store, config=Config(consolidate_similarity_threshold=0.85, consolidate_majority_k=1))
    assert r1["atoms_merged"] == 1
    r2 = consolidate(store, config=Config(consolidate_similarity_threshold=0.85, consolidate_majority_k=1))
    assert r2["atoms_merged"] == 0                 # merged atom excluded now


@patch("ccchain.core.embedding.embed")
@patch("ccchain.core.llm.chat_json_majority")
def test_parent_grouping_prevents_cross_branch_merge(mock_majority, mock_embed, store):
    """(c) two similar W3 atoms under DIFFERENT W2 parents → never merged."""
    mock_embed.side_effect = _embed_factory()
    mock_majority.side_effect = _arbiter_factory(merge=True)

    w2a = Atom(node_id="w2_a", name="Problem A", type="bottleneck",
               level="W1_problem", context="pa", provenance={})
    w2b = Atom(node_id="w2_b", name="Problem B", type="bottleneck",
               level="W1_problem", context="pb", provenance={})
    w3a = Atom(node_id="w3_x", name="OT A", type="method",
               level="W2_direction", context="OT [emb:OT]", provenance={})
    w3b = Atom(node_id="w3_y", name="OT B", type="method",
               level="W2_direction", context="OT [emb:OT]", provenance={})
    edges = [
        Edge(src="w2_a", relation="decomposes_into", tgt="w3_x"),
        Edge(src="w3_x", relation="aggregates_to", tgt="w2_a"),
        Edge(src="w2_b", relation="decomposes_into", tgt="w3_y"),
        Edge(src="w3_y", relation="aggregates_to", tgt="w2_b"),
    ]
    for a in (w3a, w3b):
        a.embedding = _embed_factory()([a.context])[0]
    store.insert_blueprint([w2a, w2b, w3a, w3b], edges, "p.pdf")

    report = consolidate(store, config=Config(consolidate_similarity_threshold=0.85, consolidate_majority_k=1))
    # same cosine but different parents → 0 candidate clusters → arbiter never called
    assert report["clusters_formed"] == 0
    assert report["atoms_merged"] == 0
    assert mock_majority.call_count == 0


@patch("ccchain.core.embedding.embed")
@patch("ccchain.core.llm.chat_json_majority")
def test_arbiter_says_no_leaves_atoms_separate(mock_majority, mock_embed, store):
    """(d) arbiter returns merge=false → atoms unchanged."""
    mock_embed.side_effect = _embed_factory()
    mock_majority.side_effect = _arbiter_factory(merge=False)
    _two_papers_ot_under_same_parent(store)

    report = consolidate(store, config=Config(consolidate_similarity_threshold=0.85, consolidate_majority_k=1))
    assert report["clusters_confirmed"] == 0
    assert report["atoms_merged"] == 0
    a = store.query_by_id("w3_a")
    b = store.query_by_id("w3_b")
    assert a.status != "merged" and b.status != "merged"
    assert a.name == "OT Method A"                 # unchanged


@patch("ccchain.core.embedding.embed")
@patch("ccchain.core.llm.chat_json_majority")
def test_consolidate_rewires_child_edges(mock_majority, mock_embed, store):
    """Merged-away atom's W4 child edge rewires to the canonical."""
    mock_embed.side_effect = _embed_factory()
    mock_majority.side_effect = _arbiter_factory(merge=True)
    _two_papers_ot_under_same_parent(store)
    # add a W4 child of w3_b
    w4 = Atom(node_id="w4_child", name="Sinkhorn", type="solution",
              level="W4_implementation", context="sinkhorn impl", provenance={})
    store.insert_blueprint([w4], [Edge(src="w3_b", relation="decomposes_into", tgt="w4_child")], "p.pdf")

    consolidate(store, config=Config(consolidate_similarity_threshold=0.85, consolidate_majority_k=1))

    # after merge, the canonical w3_a neighbors the child; merged w3_b does not
    assert "w4_child" in set(store.get_neighbors("w3_a"))
    assert "w4_child" not in set(store.get_neighbors("w3_b"))
