"""Interactive HTML visualization of CoE audit outcomes.

Public entry point:

    from ccchain.visualize import build_audit_html
    path = build_audit_html(store, reports, "audit_report.html")

Renders the knowledge graph with nodes whose FILL encodes audit status and
whose BORDER encodes the W2→W3→W4→W5 specification level, plus per-paper CPR
cards and a click-through detail panel showing each atom's CoE check verdicts.

This is a reporting utility, not one of the three SDK methods (ingest/search/
evaluate): it consumes a CCStore and one or more audit reports and emits a
self-contained .html file (single dependency: the vis-network CDN script).
"""

from __future__ import annotations

import html
import json
import os

# Status → color palette (dark-theme friendly)
STATUS_COLORS = {
    "verified":        {"bg": "#10b981", "border": "#34d399", "label": "Verified"},
    "skipped":         {"bg": "#64748b", "border": "#94a3b8", "label": "Skipped"},
    "low_confidence":  {"bg": "#f59e0b", "border": "#fbbf24", "label": "Low Confidence"},
    "low_reliability": {"bg": "#ef4444", "border": "#f87171", "label": "Low Reliability"},
    "demoted":         {"bg": "#a855f7", "border": "#c084fc", "label": "Demoted"},
    "active":          {"bg": "#3b82f6", "border": "#60a5fa", "label": "Active"},
    "needs_review":    {"bg": "#eab308", "border": "#facc15", "label": "Needs Review"},
}

LEVEL_LABEL = {
    "W2_problem_analysis": "W2",
    "W3_solution_direction": "W3",
    "W4_concrete_solution": "W4",
    "W5_code_implementation": "W5",
}
LEVEL_ORDER = {0: "W2_problem_analysis", 1: "W3_solution_direction",
               2: "W4_concrete_solution", 3: "W5_code_implementation"}
LEVEL_TO_INT = {v: k for k, v in LEVEL_ORDER.items()}
_LEVELS = list(LEVEL_ORDER.values())

# Level → border color + semantic role (specification hierarchy W2→W3→W4→W5)
LEVEL_BORDER = {
    "W2_problem_analysis":   {"border": "#ef4444", "label": "W2 Problem"},
    "W3_solution_direction": {"border": "#f59e0b", "label": "W3 Direction"},
    "W4_concrete_solution":  {"border": "#3b82f6", "label": "W4 Solution"},
    "W5_code_implementation":{"border": "#10b981", "label": "W5 Code"},
}

CHECK_LABELS = {
    "I1": "Score Verification",
    "I2": "Specification Violation",
    "I3": "Reference Verification",
    "I4": "Method-Code Alignment",
}

# Node size by level (W2 largest → W5 smallest), reinforces the pyramid
_LEVEL_SIZE = {
    "W2_problem_analysis": 26, "W3_solution_direction": 18,
    "W4_concrete_solution": 13, "W5_code_implementation": 9,
}

# Edge styling by relation
EDGE_STYLES = {
    "decomposes_into": {"color": "#3b82f6", "dashes": True, "width": 1.0},
    "aggregates_to":   {"color": "#f59e0b", "dashes": False, "width": 1.4},
    "extends":         {"color": "#10b981", "dashes": False, "width": 0.8},
    "improves":        {"color": "#34d399", "dashes": False, "width": 0.8},
    "compares":        {"color": "#a78bfa", "dashes": True, "width": 0.6},
    "uses_component":  {"color": "#64748b", "dashes": False, "width": 0.6},
}


def build_audit_html(
    store,
    reports,
    output_path: str,
    *,
    title: str = "ccchain · CoE Audit Report",
) -> str:
    """Render an interactive CoE audit report to a self-contained HTML file.

    Args:
        store: a :class:`~ccchain.core.store.CCStore`. Queried for every atom
            (all levels, all statuses) and for the igraph edges.
        reports: audit reports to summarize in the cards. Each item is either a
            ``(label, audit_report)`` pair or a ``(label, ingest_result,
            audit_report)`` triple — both accepted for convenience. ``label`` is
            the source filename/identifier shown on the card; ``audit_report``
            is the dict returned by the CoE verifier (keys: cpr, atoms_audited,
            atoms_passed, atoms_failed, atoms_skipped, failures_by_check,
            per_atom).
        output_path: destination ``.html`` path (parent dirs are created).
        title: HTML ``<title>`` and header label.

    Returns:
        The absolute path of the written file.
    """
    normalized = [_normalize_report(r) for r in (reports or [])]

    # ── per-atom check map + nodes ──────────────────────────────────────
    check_map: dict[str, dict] = {}
    for _label, ar in normalized:
        for pa in ar.get("per_atom", []):
            check_map[pa["node_id"]] = pa.get("checks", {}) or {}

    atoms = []
    for lvl in _LEVELS:
        atoms.extend(store.query_by_level(lvl, status=None))

    nodes = []
    atom_meta: dict[str, dict] = {}
    for a in atoms:
        status = a.status or "active"
        short = (a.name or a.node_id)[:42]
        nodes.append({
            "id": a.node_id,
            "label": short,
            "group": status,
            "level": LEVEL_TO_INT.get(a.level, 2),
            "cc_level": a.level,
            "type": a.type,
            "status": status,
            "size": _LEVEL_SIZE.get(a.level, 10),
            "title": f"[{LEVEL_LABEL.get(a.level, a.level)}] {short}\n{a.type} · {status}",
        })
        atom_meta[a.node_id] = {
            "name": a.name,
            "type": a.type,
            "level": a.level,
            "status": status,
            "source_pdf": a.source_pdf or "",
            "context": a.context or "",
            "provenance": a.provenance or {},
            "checks": check_map.get(a.node_id, {}),
        }

    # ── edges from the igraph store ─────────────────────────────────────
    g = store.graph
    edges = []
    for e in g.es:
        try:
            src_name = g.vs[e.source]["name"]
            tgt_name = g.vs[e.target]["name"]
        except (IndexError, KeyError):
            continue
        rel = e["relation"] or "related_to"
        st = EDGE_STYLES.get(rel, {"color": "#475569", "dashes": False, "width": 0.5})
        edges.append({
            "from": src_name, "to": tgt_name, "label": rel, "relation": rel,
            "color": {"color": st["color"], "highlight": st["color"], "hover": st["color"]},
            "dashes": st["dashes"], "width": st["width"],
        })

    # ── per-paper cards + legend + toggles ──────────────────────────────
    cards = [_card_data(label, ar) for label, ar in normalized]
    cards_html = _render_cards(cards)

    status_legend = "".join(
        f'<div class="legend-item"><div class="dot" style="background:{c["bg"]}"></div>{c["label"]}</div>'
        for c in STATUS_COLORS.values()
    )
    level_legend_items = "".join(
        f'<div class="legend-item"><div class="ring" style="border-color:{LEVEL_BORDER[lvl]["border"]}"></div>'
        f'{LEVEL_BORDER[lvl]["label"]}</div>'
        for lvl in _LEVELS
    )
    legend_html = (
        f'<div class="legend-group"><span class="lg-title">fill=status</span>{status_legend}</div>'
        f'<div class="legend-group"><span class="lg-title">border=level (spec chain W2→W3→W4→W5)</span>{level_legend_items}</div>'
    )
    status_present = sorted({n["status"] for n in nodes})
    toggle_buttons = "".join(
        f'<button class="btn status-btn" data-status="{s}" '
        f'style="border-color:{STATUS_COLORS[s]["border"]};color:{STATUS_COLORS[s]["border"]}">'
        f'{STATUS_COLORS[s]["label"]}</button>'
        for s in status_present
    )

    payload = {
        "nodes": nodes,
        "edges": edges,
        "atom_meta": atom_meta,
        "check_labels": CHECK_LABELS,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    full = _HTML_TEMPLATE.format(
        title=html.escape(title),
        cards_html=cards_html,
        legend_html=legend_html,
        toggle_buttons=toggle_buttons,
        node_count=len(nodes),
        edge_count=len(edges),
        paper_count=len(cards),
        payload=json.dumps(payload),
        status_palette_json=json.dumps({k: v["bg"] for k, v in STATUS_COLORS.items()}),
        level_border_json=json.dumps({k: v["border"] for k, v in LEVEL_BORDER.items()}),
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_report(item) -> tuple[str, dict]:
    """Accept (label, audit) or (label, result, audit) → (label, audit)."""
    if len(item) == 3:
        return str(item[0]), item[2]
    if len(item) == 2:
        return str(item[0]), item[1]
    raise ValueError(
        f"Each report must be a (label, audit_report) pair or "
        f"(label, ingest_result, audit_report) triple; got {len(item)}-tuple."
    )


def _card_data(label: str, ar: dict) -> dict:
    from collections import Counter
    status_counts = Counter(pa["status"] for pa in ar.get("per_atom", []))
    total = sum(status_counts.values()) or 1
    return {
        "filename": label,
        "cpr": ar.get("cpr", 0.0),
        "atoms_audited": ar.get("atoms_audited", 0),
        "atoms_passed": ar.get("atoms_passed", 0),
        "atoms_failed": ar.get("atoms_failed", 0),
        "atoms_skipped": ar.get("atoms_skipped", 0),
        "failures": ar.get("failures_by_check", {"I1": 0, "I2": 0, "I3": 0, "I4": 0}),
        "status_counts": dict(status_counts),
        "status_total": total,
    }


def _render_cards(cards: list[dict]) -> str:
    if not cards:
        return '<div class="empty">No audit reports.</div>'
    parts = []
    for c in cards:
        bar_segments = []
        for s in ["verified", "low_confidence", "low_reliability", "demoted", "skipped", "active"]:
            n = c["status_counts"].get(s, 0)
            if n == 0:
                continue
            pct = (n / c["status_total"]) * 100
            col = STATUS_COLORS.get(s, STATUS_COLORS["active"])["bg"]
            bar_segments.append(
                f'<div class="bar-seg" style="width:{pct:.1f}%;background:{col}" '
                f'title="{STATUS_COLORS[s]["label"]}: {n}"></div>'
            )
        bar = "".join(bar_segments) or '<div class="bar-seg" style="width:100%;background:#1e293b"></div>'

        chips = []
        for chk in ["I1", "I2", "I3", "I4"]:
            n = c["failures"].get(chk, 0)
            cls = "chip chip-fail" if n else "chip chip-ok"
            chips.append(f'<span class="{cls}">{chk}: {n}</span>')
        chips_html = "".join(chips)

        cpr_col = "#10b981" if c["cpr"] >= 0.8 else ("#f59e0b" if c["cpr"] >= 0.5 else "#ef4444")
        parts.append(f"""
        <div class="card">
          <div class="card-title">{html.escape(c["filename"])}</div>
          <div class="cpr-row">
            <div class="cpr-num" style="color:{cpr_col}">{c["cpr"]:.2f}</div>
            <div class="cpr-label">CPR<br><span class="muted">Claim Provenance Rate</span></div>
          </div>
          <div class="bar">{bar}</div>
          <div class="card-stats">
            <span>audited {c["atoms_audited"]}</span>
            <span class="ok-text">passed {c["atoms_passed"]}</span>
            <span class="fail-text">failed {c["atoms_failed"]}</span>
            <span class="muted">skipped {c["atoms_skipped"]}</span>
          </div>
          <div class="chips">{chips_html}</div>
        </div>""")
    return "".join(parts)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; }}
#header {{ padding: 14px 24px; background: linear-gradient(135deg,#1e293b,#0f172a); border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }}
h1 {{ font-size: 20px; color: #60a5fa; }}
h1 .sub {{ font-size: 12px; color: #94a3b8; font-weight: 400; }}
.stat {{ font-size: 12px; color: #94a3b8; background: #0f172a; padding: 4px 11px; border-radius: 6px; border: 1px solid #1e293b; }}
#cards {{ display: flex; gap: 14px; padding: 16px 24px; overflow-x: auto; background: #0f172a; border-bottom: 1px solid #1e293b; }}
.card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px 16px; min-width: 230px; flex-shrink: 0; }}
.card-title {{ font-size: 12px; color: #cbd5e1; font-weight: 600; margin-bottom: 10px; word-break: break-all; }}
.cpr-row {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 10px; }}
.cpr-num {{ font-size: 30px; font-weight: 700; line-height: 1; }}
.cpr-label {{ font-size: 10px; color: #94a3b8; line-height: 1.3; }}
.muted {{ color: #64748b; }}
.ok-text {{ color: #34d399; }}
.fail-text {{ color: #f87171; }}
.bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #0f172a; margin-bottom: 8px; }}
.bar-seg {{ height: 100%; transition: width .3s; }}
.card-stats {{ display: flex; gap: 10px; font-size: 10px; color: #94a3b8; flex-wrap: wrap; margin-bottom: 8px; }}
.chips {{ display: flex; gap: 5px; flex-wrap: wrap; }}
.chip {{ font-size: 9px; padding: 2px 7px; border-radius: 4px; font-weight: 600; }}
.chip-ok {{ background: #064e3b; color: #34d399; }}
.chip-fail {{ background: #4c0519; color: #f87171; }}
#controls {{ padding: 8px 24px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.btn {{ padding: 5px 13px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #cbd5e1; cursor: pointer; font-size: 11px; transition: all .15s; }}
.btn:hover {{ background: #334155; }}
.btn.active {{ filter: brightness(1.3); }}
#legend {{ display: flex; gap: 12px; align-items: center; margin-left: auto; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 10px; color: #94a3b8; white-space: nowrap; }}
.legend-group {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 0 8px; border-left: 1px solid #334155; }}
.legend-group:first-child {{ border-left: none; }}
.lg-title {{ font-size: 9px; text-transform: uppercase; letter-spacing: .05em; color: #64748b; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.ring {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; border: 2px solid; background: transparent; }}
#wrapper {{ display: flex; height: calc(100vh - 290px); min-height: 420px; }}
#mynetwork {{ flex: 1; background: #0f172a; }}
#info-panel {{ width: 360px; background: #1e293b; border-left: 1px solid #334155; padding: 18px; overflow-y: auto; font-size: 13px; line-height: 1.6; }}
#info-panel h3 {{ color: #60a5fa; font-size: 14px; margin-bottom: 4px; word-break: break-word; }}
#info-panel .meta {{ color: #94a3b8; font-size: 11px; margin: 3px 0; }}
.tag {{ display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 10px; font-weight: 600; margin-right: 4px; margin-bottom: 3px; }}
.section-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: #64748b; margin: 12px 0 4px; }}
.check-row {{ display: flex; align-items: center; gap: 8px; padding: 5px 8px; border-radius: 6px; background: #0f172a; margin-bottom: 4px; font-size: 11px; }}
.check-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.check-name {{ font-weight: 600; color: #cbd5e1; min-width: 28px; }}
.check-status {{ font-size: 10px; padding: 1px 6px; border-radius: 3px; }}
.st-pass {{ background: #064e3b; color: #34d399; }}
.st-fail {{ background: #4c0519; color: #f87171; }}
.st-skip {{ background: #1e293b; color: #64748b; }}
.body-box {{ color: #cbd5e1; font-size: 11px; background: #0f172a; padding: 8px; border-radius: 6px; max-height: 110px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}
.empty {{ color: #64748b; font-style: italic; padding: 20px; }}
footer {{ position: fixed; bottom: 4px; left: 12px; color: #334155; font-size: 10px; }}
</style>
</head>
<body>
<div id="header">
  <h1>{title} <span class="sub">Chain-of-Evidence integrity (I1/I2/I3/I4)</span></h1>
  <span class="stat">Papers: {paper_count}</span>
  <span class="stat">Atoms: {node_count}</span>
  <span class="stat">Edges: {edge_count}</span>
</div>
<div id="cards">{cards_html}</div>
<div id="controls">
  <span class="muted" style="font-size:10px">Filter by status:</span>
  {toggle_buttons}
  <button class="btn" onclick="selectAll()">Show All</button>
  <button class="btn" onclick="fitGraph()">Fit</button>
  <div id="legend">{legend_html}</div>
</div>
<div id="wrapper">
  <div id="mynetwork"></div>
  <div id="info-panel">
    <h3>Atom Detail</h3>
    <div class="meta muted">Click a node to inspect its CoE check verdicts.</div>
    <div id="info-content"></div>
  </div>
</div>
<footer>ccchain v0.3 · CoE audit</footer>

<script>
const data = {payload};
const statusColor = {status_palette_json};
const levelBorder = {level_border_json};
const checkLabels = data.check_labels;

const nodes = new vis.DataSet(data.nodes.map(n => ({{
  ...n,
  borderWidth: 3,
  color: {{ background: statusColor[n.status] || '#3b82f6',
            border: levelBorder[n.cc_level] || '#0f172a',
            highlight: {{ background: statusColor[n.status] || '#3b82f6', border: '#f1f5f9' }},
            hover: {{ background: statusColor[n.status] || '#3b82f6', border: '#e2e8f0' }} }},
}})));
const edges = new vis.DataSet(data.edges);

const options = {{
  nodes: {{ shape: 'dot', font: {{ size: 9, color: '#cbd5e1', strokeWidth: 2, strokeColor: '#0f172a' }},
            borderWidth: 1, borderWidthSelected: 3, shadow: {{ enabled: true, color: 'rgba(0,0,0,.5)', size: 6 }} }},
  edges: {{ font: {{ size: 7, color: '#475569', strokeWidth: 2, strokeColor: '#0f172a' }},
            arrows: {{ to: {{ enabled: true, scaleFactor: 0.4 }} }}, smooth: {{ type: 'curvedCW', roundness: 0.12 }} }},
  physics: {{ enabled: true, solver: 'hierarchicalRepulsion',
              hierarchicalRepulsion: {{ centralGravity: 0.0, springLength: 180, springConstant: 0.01, nodeDistance: 130, damping: 0.09 }},
              stabilization: {{ iterations: 200 }} }},
  layout: {{ hierarchical: {{ enabled: true, direction: 'UD', sortMethod: 'directed',
              levelSeparation: 200, nodeSpacing: 150, treeSpacing: 40, blockShifting: true, edgeMinimization: true }} }},
  interaction: {{ hover: true, tooltipDelay: 120, zoomView: true, dragView: true }},
}};
const network = new vis.Network(document.getElementById('mynetwork'), {{ nodes, edges }}, options);

// ── Status filter ────────────────────────────────────────────────────
const hiddenStatus = new Set();
document.querySelectorAll('.status-btn').forEach(btn => {{
  btn.classList.add('active');
  btn.addEventListener('click', () => {{
    const s = btn.dataset.status;
    if (hiddenStatus.has(s)) {{ hiddenStatus.delete(s); btn.classList.add('active'); }}
    else {{ hiddenStatus.add(s); btn.classList.remove('active'); }}
    data.nodes.forEach(n => {{
      nodes.update({{ id: n.id, hidden: hiddenStatus.has(n.status) }});
    }});
  }});
}});
function selectAll() {{
  hiddenStatus.clear();
  document.querySelectorAll('.status-btn').forEach(b => b.classList.add('active'));
  data.nodes.forEach(n => nodes.update({{ id: n.id, hidden: false }}));
}}
function fitGraph() {{ network.fit({{ animation: true }}); }}

// ── Click → detail panel ─────────────────────────────────────────────
network.on('click', params => {{
  if (params.nodes.length === 0) return;
  const nid = params.nodes[0];
  const meta = data.atom_meta[nid];
  if (!meta) return;
  const c = document.getElementById('info-content');
  const lvlTag = '<span class="tag" style="background:#334155;color:#cbd5e1">' + meta.level + '</span>';
  const typeTag = '<span class="tag" style="background:#1e293b;color:#94a3b8">' + meta.type + '</span>';
  const stCol = statusColor[meta.status] || '#3b82f6';
  const stTag = '<span class="tag" style="background:' + stCol + ';color:#0f172a">' + meta.status + '</span>';
  let h = '<div style="margin-bottom:6px">' + lvlTag + typeTag + stTag + '</div>';
  h += '<h3>' + (meta.name || nid) + '</h3>';
  if (meta.source_pdf) h += '<div class="meta">source: ' + meta.source_pdf + '</div>';
  h += '<div class="section-label">Context</div><div class="body-box">' + esc(meta.context) + '</div>';
  const chk = meta.checks || {{}};
  const keys = Object.keys(chk);
  if (keys.length) {{
    h += '<div class="section-label">CoE Checks</div>';
    keys.sort().forEach(k => {{
      const r = chk[k];
      const st = r.status || 'skipped';
      const col = st === 'passed' ? '#10b981' : (st === 'failed' ? '#ef4444' : '#64748b');
      const cls = st === 'passed' ? 'st-pass' : (st === 'failed' ? 'st-fail' : 'st-skip');
      h += '<div class="check-row"><div class="check-dot" style="background:' + col + '"></div>' +
           '<span class="check-name">' + k + '</span>' +
           '<span class="check-status ' + cls + '">' + st + '</span>' +
           '<span style="color:#64748b;font-size:10px">' + esc(checkLabels[k] || '') + '</span></div>';
      const detail = r.reasoning || r.reason || '';
      if (detail) h += '<div class="meta" style="margin-left:16px">' + esc(String(detail)) + '</div>';
    }});
  }} else {{
    h += '<div class="section-label">CoE Checks</div><div class="meta muted">None applicable for this atom type.</div>';
  }}
  const pv = meta.provenance || {{}};
  if (Object.keys(pv).length) {{
    h += '<div class="section-label">Provenance</div><div class="body-box">' + esc(JSON.stringify(pv, null, 1)) + '</div>';
  }}
  c.innerHTML = h;
}});

function esc(s) {{ return String(s).replace(/[&<>]/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[m])); }}
</script>
</body>
</html>
"""
