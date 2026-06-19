"""Test ccchain.okf — OKF v0.1 export/import interoperability."""

import os
import tempfile

import pytest
import yaml

from ccchain.core.ontology import Atom, Edge
from ccchain.core.store import CCStore
from ccchain.okf import export_okf, import_okf


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp(prefix="ccchain_okf_")
    s = CCStore(db_path=os.path.join(tmp, "t.db"), graph_dir=tmp)
    atoms = [
        Atom(node_id="w1", name="RL Problem", type="bottleneck",
             level="W1_problem", context="credit assignment noise",
             source_pdf="cop-q.pdf", tags=["domain:MARL"],
             provenance={"phase": "extract"}),
        Atom(node_id="w2", name="OT Direction", type="method",
             level="W2_direction", context="OT-based credit",
             source_pdf="cop-q.pdf", provenance={"code_span": "sec 2"}),
        Atom(node_id="w3", name="Sinkhorn Approach", type="method",
             level="W3_approach", context="entropy-reg OT",
             source_pdf="cop-q.pdf", provenance={"code_span": "sec 3"}),
        Atom(node_id="w4", name="Win Rate 0.9", type="numerical",
             level="W4_implementation", context="win rate 0.9 on SMAC",
             source_pdf="cop-q.pdf", provenance={"score": 0.9}),
        Atom(node_id="w5", name="sinkhorn()", type="component",
             level="W5_code", context="solver wrapper",
             source_pdf="cop-q.pdf",
             code_ref="sinkhorn", code_body="def sinkhorn(c, reg=0.1): ..."),
    ]
    edges = [
        Edge(src="w1", relation="decomposes_into", tgt="w2"),
        Edge(src="w2", relation="decomposes_into", tgt="w3"),
        Edge(src="w3", relation="decomposes_into", tgt="w4"),
        Edge(src="w4", relation="decomposes_into", tgt="w5"),
        Edge(src="w3", relation="aggregates_to", tgt="w2"),
    ]
    s.insert_blueprint(atoms, edges, "cop-q.pdf")
    yield s
    s.db.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_export_creates_bundle_structure(store, tmp_path):
    bundle = export_okf(store, str(tmp_path / "bundle"))
    assert os.path.isdir(bundle)
    # one concept file per atom + index.md + log.md
    md_files = []
    for dp, _, fns in os.walk(bundle):
        for fn in fns:
            if fn.endswith(".md"):
                md_files.append(fn)
    assert "index.md" in md_files
    assert "log.md" in md_files
    # 5 concept files (one per atom), levels as subdirs
    for lvl in ["W1_problem", "W2_direction", "W3_approach",
                "W4_implementation", "W5_code"]:
        assert os.path.isdir(os.path.join(bundle, lvl))


def test_exported_concept_is_okf_conformant(store, tmp_path):
    """Every concept has the required `type` frontmatter + markdown body."""
    bundle = export_okf(store, str(tmp_path / "bundle"))
    concept_paths = []
    for dp, _, fns in os.walk(bundle):
        for fn in fns:
            if fn.endswith(".md") and fn not in ("index.md", "log.md"):
                concept_paths.append(os.path.join(dp, fn))
    assert len(concept_paths) == 5
    for p in concept_paths:
        text = open(p, encoding="utf-8").read()
        assert text.startswith("---")
        fm, _ = text[3:].split("\n---", 1)
        meta = yaml.safe_load(fm)
        assert "type" in meta  # the ONE required OKF field
        assert "title" in meta and "description" in meta


def test_export_carries_ccchain_state(store, tmp_path):
    """ccchain extensions (level/status/provenance) survive into frontmatter."""
    bundle = export_okf(store, str(tmp_path / "bundle"))
    w4 = open(os.path.join(bundle, "W4_implementation", "w4.md"), encoding="utf-8").read()
    fm = yaml.safe_load(w4[3:].split("\n---", 1)[0])
    assert fm["type"] == "numerical"
    assert fm["ccchain_level"] == "W4_implementation"
    assert fm["ccchain_node_id"] == "w4"
    assert fm["ccchain_provenance"]["score"] == 0.9


def test_export_edges_become_markdown_links(store, tmp_path):
    bundle = export_okf(store, str(tmp_path / "bundle"))
    w3 = open(os.path.join(bundle, "W3_approach", "w3.md"), encoding="utf-8").read()
    # outgoing decomposes_into → w4, incoming aggregates_to ← w2
    assert "decomposes_into" in w3 and "](W4_implementation/w4.md)" in w3
    assert "aggregates_to" in w3 and "](W2_direction/w2.md)" in w3


def test_export_w5_includes_code_block(store, tmp_path):
    bundle = export_okf(store, str(tmp_path / "bundle"))
    w5 = open(os.path.join(bundle, "W5_code", "w5.md"), encoding="utf-8").read()
    assert "```python" in w5 and "def sinkhorn" in w5


# ---------------------------------------------------------------------------
# Round-trip: export → import → same graph
# ---------------------------------------------------------------------------
def test_roundtrip_preserves_atoms_and_edges(store, tmp_path):
    """Export then import into a fresh store recovers atoms + edges."""
    bundle = export_okf(store, str(tmp_path / "bundle"))

    tmp2 = tempfile.mkdtemp(prefix="ccchain_okf_rt_")
    store2 = CCStore(db_path=os.path.join(tmp2, "t.db"), graph_dir=tmp2)
    try:
        report = import_okf(bundle, store2)
        assert report["atoms_imported"] == 5
        assert report["edges_imported"] >= 4  # decomposes chain (some dups dropped)
        # all 5 node_ids present
        for nid in ["w1", "w2", "w3", "w4", "w5"]:
            assert store2.query_by_id(nid) is not None
        # atom content preserved
        w4 = store2.query_by_id("w4")
        assert w4.type == "numerical"
        assert w4.level == "W4_implementation"
        assert w4.provenance["score"] == 0.9
        # edge preserved
        assert "w2" in store2.get_neighbors("w1")  # w1 decomposes_into w2
    finally:
        store2.db.close()


# ---------------------------------------------------------------------------
# Import: external (non-ccchain) OKF bundle
# ---------------------------------------------------------------------------
def test_import_external_okf_bundle(tmp_path):
    """A hand-written OKF bundle (no ccchain_* fields) imports gracefully."""
    bundle = tmp_path / "external"
    (bundle / "concepts").mkdir(parents=True)
    # external producer uses its own `type` vocabulary + only required fields
    (bundle / "concepts" / "metric.md").write_text(
        "---\n"
        "type: metric\n"
        "title: Daily Active Users\n"
        "description: Count of distinct users active in a day.\n"
        "resource: events_table\n"
        "tags: [product]\n"
        "---\n\n"
        "Count of distinct users active in a day.\n",
        encoding="utf-8",
    )
    (bundle / "concepts" / "table.md").write_text(
        "---\n"
        "type: table\n"
        "title: events\n"
        "description: Raw event stream.\n"
        "---\n\n"
        "Raw event stream.\n\n"
        "## Relations\n"
        "- related_to → [Daily Active Users](metric.md)\n",
        encoding="utf-8",
    )

    tmp2 = tempfile.mkdtemp(prefix="ccchain_okf_ext_")
    store = CCStore(db_path=os.path.join(tmp2, "t.db"), graph_dir=tmp2)
    try:
        report = import_okf(str(bundle), store)
        assert report["atoms_imported"] == 2
        assert report["edges_imported"] == 1
        # unknown 'metric'/'table' types coerced to a valid ccchain type, original recorded
        atoms = {a.name: a for l in
                 ["W1_problem", "W2_direction", "W3_approach", "W4_implementation", "W5_code"]
                 for a in store.query_by_level(l, status=None)}
        assert "Daily Active Users" in atoms
        da = atoms["Daily Active Users"]
        assert da.provenance.get("okf_original_type") == "metric"
    finally:
        store.db.close()
