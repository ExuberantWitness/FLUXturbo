"""Test ccchain.visualize.build_audit_html — standalone HTML report module."""

import os
import tempfile

import pytest

from ccchain.core.ontology import Atom, Edge
from ccchain.core.store import CCStore
from ccchain.visualize import build_audit_html


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp(prefix="ccchain_vis_")
    s = CCStore(db_path=os.path.join(tmp, "t.db"), graph_dir=tmp)
    atoms = [
        Atom(node_id="w2", name="Problem", type="bottleneck", level="W2_problem_analysis",
             context="core problem", source_pdf="p.pdf",
             provenance={"phase": "test"}),
        Atom(node_id="w3", name="Method", type="method", level="W3_solution_direction",
             context="an OT method", source_pdf="p.pdf",
             provenance={"code_span": "sec 3"}),
        Atom(node_id="w4", name="Win rate", type="numerical", level="W4_concrete_solution",
             context="win rate 0.9", source_pdf="p.pdf",
             provenance={"score": 0.9, "score_std": 0.01}),
        Atom(node_id="w5", name="impl", type="component", level="W5_code_implementation",
             context="code", source_pdf="p.pdf"),
    ]
    edges = [
        Edge(src="w2", relation="decomposes_into", tgt="w3"),
        Edge(src="w3", relation="decomposes_into", tgt="w4"),
        Edge(src="w4", relation="decomposes_into", tgt="w5"),
        Edge(src="w4", relation="aggregates_to", tgt="w3"),
    ]
    s.insert_blueprint(atoms, edges, "p.pdf")
    yield s
    s.db.close()


_AUDIT_REPORT = {
    "cpr": 1.0,
    "atoms_audited": 2,
    "atoms_passed": 1,
    "atoms_failed": 1,
    "atoms_skipped": 2,
    "failures_by_check": {"I1": 0, "I2": 0, "I3": 1, "I4": 0},
    "per_atom": [
        {"node_id": "w3", "type": "method", "status": "verified",
         "checks": {"I4": {"status": "passed", "verdict": "aligned"}}},
        {"node_id": "w4", "type": "numerical", "status": "low_reliability",
         "checks": {"I1": {"status": "failed"}}},
        {"node_id": "w2", "type": "bottleneck", "status": "skipped", "checks": {}},
        {"node_id": "w5", "type": "component", "status": "skipped", "checks": {}},
    ],
}


def test_build_audit_html_writes_file_and_returns_abs_path(store, tmp_path):
    out = tmp_path / "report.html"
    abs_path = build_audit_html(store, [("p.pdf", _AUDIT_REPORT)], str(out))
    assert os.path.isabs(abs_path)
    assert os.path.exists(abs_path)
    html = out.read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")


def test_build_audit_html_accepts_triple_reports(store, tmp_path):
    """3-tuple (label, ingest_result, audit) must also work (backward-compat)."""
    out = tmp_path / "r.html"
    fake_result = {"node_count_by_level": {}, "edge_count": 4}
    build_audit_html(store, [("p.pdf", fake_result, _AUDIT_REPORT)], str(out))
    html = out.read_text(encoding="utf-8")
    # card label rendered
    assert "p.pdf" in html


def test_build_audit_html_embeds_payload_and_legend(store, tmp_path):
    out = tmp_path / "r.html"
    build_audit_html(store, [("p.pdf", _AUDIT_REPORT)], str(out))
    html = out.read_text(encoding="utf-8")
    # status palette + level border both injected
    assert "const statusColor" in html
    assert "const levelBorder" in html
    # spec-chain legend present
    assert "border=level" in html
    # all 4 levels' border colors referenced
    for lvl_label in ["W2 Problem", "W3 Direction", "W4 Solution", "W5 Code"]:
        assert lvl_label in html
    # node ids present in the JSON payload
    assert '"w4"' in html


def test_build_audit_html_cpr_card_content(store, tmp_path):
    out = tmp_path / "r.html"
    build_audit_html(store, [("my-paper", _AUDIT_REPORT)], str(out))
    html = out.read_text(encoding="utf-8")
    assert "my-paper" in html
    assert "1.00" in html  # CPR rendered
    # failure chip for I3
    assert "I3: 1" in html


def test_build_audit_html_creates_parent_dir(store, tmp_path):
    out = tmp_path / "nested" / "deep" / "r.html"
    build_audit_html(store, [("p.pdf", _AUDIT_REPORT)], str(out))
    assert out.exists()


def test_build_audit_html_empty_reports(store, tmp_path):
    out = tmp_path / "empty.html"
    build_audit_html(store, [], str(out))
    html = out.read_text(encoding="utf-8")
    assert "No audit reports." in html
