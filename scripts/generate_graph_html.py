"""Generate interactive vis-network HTML from the ccchain knowledge graph.

Reads from the ccchain SQLite + igraph store and produces a dark-themed,
interactive graph visualization showing the W2→W3→W4→W5 pyramid.

Usage:
    cd E:\DATA\vscode\FLUXturbo
    python scripts/generate_graph_html.py              # from existing store
    python scripts/generate_graph_html.py --sample     # quick sample data first

Requires: igraph, sqlite3 (both already in ccchain deps)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blueprint_output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cc_graph.html")


def generate_html():
    from ccchain.config import Config
    from ccchain.core.store import CCStore

    config = Config()
    store = CCStore(config.db_path, config.graph_dir)
    g = store.graph

    if g.vcount() == 0:
        print("Store is empty. Run with --sample to add demo data, or run e2e_live_test.py first.")
        return None

    print(f"Graph: {g.vcount()} nodes, {g.ecount()} edges")

    # ── Build node/edge JSON ────────────────────────────────────────────
    nodes = []
    level_counts = {}
    level_to_int = {
        "W1_problem": 0,
        "W2_direction": 1,
        "W4_implementation": 2,
        "W5_code": 3,
    }

    for v in g.vs:
        lvl = v["level"] or "unknown"
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
        short_name = (v["label"] or v["name"])[:40]
        nodes.append({
            "id": v.index,
            "label": short_name,
            "group": lvl,        # vis-network group = ccchain level (for color/size)
            "level": level_to_int.get(lvl, 2),  # integer level for hierarchical layout
            "cc_level": lvl,     # original ccchain level string
            "type": v["type"] or "unknown",
            "name": v["name"],
            "title": f"[{lvl}] {short_name}\n{v['type'] or 'unknown'}",
        })

    edges = []
    edge_rels = {}
    for e in g.es:
        rel = e["relation"] or "related_to"
        edge_rels[rel] = edge_rels.get(rel, 0) + 1
        edges.append({
            "from": e.source,
            "to": e.target,
            "label": rel,
            "relation": rel,
        })

    print(f"  Levels: {level_counts}")
    print(f"  Edge types: {edge_rels}")

    store.db.close()

    # ── Level colors ────────────────────────────────────────────────────
    level_colors = {
        "W1_problem":     {"bg": "#ef4444", "border": "#f87171", "highlight": "#dc2626", "label": "W2 Problem"},
        "W2_direction":   {"bg": "#f59e0b", "border": "#fbbf24", "highlight": "#d97706", "label": "W3 Direction"},
        "W4_implementation":    {"bg": "#3b82f6", "border": "#60a5fa", "highlight": "#2563eb", "label": "W4 Solution"},
        "W5_code":  {"bg": "#10b981", "border": "#34d399", "highlight": "#059669", "label": "W5 Code"},
    }

    # ── Build HTML ──────────────────────────────────────────────────────
    groups_json = {}
    for lvl, c in level_colors.items():
        groups_json[lvl] = {
            "color": {"background": c["bg"], "border": c["border"],
                       "highlight": {"background": c["highlight"], "border": c["border"]}},
            "shape": "box" if lvl == "W1_problem" else "dot",
            "size": {"W1_problem": 24, "W2_direction": 16,
                     "W4_implementation": 12, "W5_code": 8}.get(lvl, 10),
            "font": {"size": 9, "color": "#cbd5e1", "face": "sans-serif",
                     "strokeWidth": 2, "strokeColor": "#0f172a"},
        }

    # Edge styles by relation
    edge_styles = {
        "aggregates_to":    {"color": "#f59e0b", "dashes": False, "width": 1.5},
        "decomposes_into":  {"color": "#3b82f6", "dashes": True,  "width": 1.0},
        "extends":          {"color": "#10b981", "dashes": False, "width": 0.8},
        "improves":         {"color": "#34d399", "dashes": False, "width": 0.8},
        "replaces":         {"color": "#ef4444", "dashes": False, "width": 0.8},
        "compares":         {"color": "#a78bfa", "dashes": True,  "width": 0.6},
        "uses_component":   {"color": "#64748b", "dashes": False, "width": 0.6},
        "adapts":           {"color": "#fbbf24", "dashes": False, "width": 0.8},
    }

    # Enrich edges with styles
    for e in edges:
        rel = e["relation"]
        style = edge_styles.get(rel, {"color": "#475569", "dashes": False, "width": 0.5})
        e["color"] = {"color": style["color"], "highlight": style["color"], "hover": style["color"]}
        e["dashes"] = style["dashes"]
        e["width"] = style["width"]

    # Build legend
    legend_html = "".join(
        f'<div class="legend-item"><div class="dot" style="background:{c["bg"]};width:12px;height:12px"></div>{c["label"]}</div>'
        for c in level_colors.values()
    )
    toggle_buttons = "".join(
        f'<button class="btn" id="btn-{lvl}" onclick="toggleLevel(\'{lvl}\')">{c["label"]}</button>'
        for lvl, c in level_colors.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ccchain — Knowledge Graph (W2→W3→W4→W5)</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; overflow: hidden; }}
#header {{ padding: 12px 24px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
h1 {{ font-size: 18px; color: #60a5fa; white-space: nowrap; }}
.stat {{ font-size: 12px; color: #94a3b8; background: #0f172a; padding: 3px 10px; border-radius: 6px; white-space: nowrap; }}
#controls {{ padding: 8px 24px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.btn {{ padding: 5px 14px; border-radius: 6px; border: 1px solid #334155; background: #1e293b; color: #cbd5e1; cursor: pointer; font-size: 12px; white-space: nowrap; transition: all 0.15s; }}
.btn:hover {{ background: #334155; color: #f1f5f9; }}
.btn.active {{ background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }}
#search-box {{ padding: 5px 12px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; font-size: 13px; width: 220px; }}
#search-box::placeholder {{ color: #475569; }}
#legend {{ display: flex; gap: 14px; align-items: center; margin-left: auto; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: #94a3b8; white-space: nowrap; }}
.dot {{ border-radius: 50%; flex-shrink: 0; }}
#mynetwork {{ width: 100%; height: calc(100vh - 110px); background: #0f172a; }}
#info-panel {{ position: fixed; bottom: 16px; right: 16px; background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px; max-width: 420px; max-height: 360px; overflow-y: auto; display: none; font-size: 13px; line-height: 1.7; z-index: 100; box-shadow: 0 4px 24px rgba(0,0,0,0.5); }}
#info-panel h3 {{ color: #60a5fa; margin-bottom: 10px; font-size: 14px; }}
#info-panel .meta {{ color: #94a3b8; font-size: 11px; margin-bottom: 4px; }}
#info-panel .body {{ color: #cbd5e1; margin-top: 8px; white-space: pre-wrap; word-break: break-word; max-height: 160px; overflow-y: auto; }}
.level-tag {{ display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 10px; font-weight: 600; margin-right: 4px; }}
#tooltip {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%); background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px 18px; pointer-events: none; display: none; font-size: 12px; z-index: 200; box-shadow: 0 2px 12px rgba(0,0,0,0.4); max-width: 320px; }}
footer {{ position: fixed; bottom: 4px; left: 12px; color: #334155; font-size: 10px; z-index: 99; }}
</style>
</head>
<body>
<div id="header">
  <h1>ccchain Knowledge Graph</h1>
  <span class="stat">Nodes: {len(nodes)}</span>
  <span class="stat">Edges: {len(edges)}</span>
  <span class="stat">W2: {level_counts.get('W1_problem', 0)}</span>
  <span class="stat">W3: {level_counts.get('W2_direction', 0)}</span>
  <span class="stat">W4: {level_counts.get('W4_implementation', 0)}</span>
  <span class="stat">W5: {level_counts.get('W5_code', 0)}</span>
</div>
<div id="controls">
  <input type="text" id="search-box" placeholder="Search node name or context...">
  {toggle_buttons}
  <button class="btn active" id="btn-labels" onclick="toggleLabels()">Hide Labels</button>
  <button class="btn" onclick="fitGraph()">Fit</button>
  <button class="btn" onclick="togglePhysics()">Freeze</button>
  <div id="legend">{legend_html}</div>
</div>
<div id="mynetwork"></div>
<div id="info-panel"><h3>Node Info</h3><div id="info-content"></div></div>
<div id="tooltip"></div>
<footer>ccchain · vis-network · W2→W3→W4→W5 knowledge pyramid</footer>

<script>
const rawData = {json.dumps({"nodes": nodes, "edges": edges})};
const nodes = new vis.DataSet(rawData.nodes);
const edges = new vis.DataSet(rawData.edges);

const container = document.getElementById('mynetwork');
const networkData = {{ nodes, edges }};

// Build per-level node lists for toggle
const levelMap = {{}};
rawData.nodes.forEach(n => {{
  if (!levelMap[n.cc_level]) levelMap[n.cc_level] = [];
  levelMap[n.cc_level].push(n.id);
}});

const options = {{
  nodes: {{
    shape: 'dot',
    font: {{ size: 9, color: '#94a3b8', face: 'sans-serif', strokeWidth: 2, strokeColor: '#0f172a' }},
    borderWidth: 1,
    borderWidthSelected: 3,
    shadow: {{ enabled: true, color: 'rgba(0,0,0,0.5)', size: 6 }},
  }},
  edges: {{
    font: {{ size: 7, color: '#475569', strokeWidth: 2, strokeColor: '#0f172a', face: 'sans-serif' }},
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.4 }} }},
    smooth: {{ type: 'curvedCW', roundness: 0.1 }},
  }},
  physics: {{
    enabled: true,
    solver: 'hierarchicalRepulsion',
    hierarchicalRepulsion: {{
      centralGravity: 0.0,
      springLength: 200,
      springConstant: 0.01,
      nodeDistance: 150,
      damping: 0.09,
    }},
    stabilization: {{ iterations: 250 }},
  }},
  layout: {{
    hierarchical: {{
      enabled: true,
      direction: 'UD',
      sortMethod: 'directed',
      levelSeparation: 220,
      nodeSpacing: 180,
      treeSpacing: 40,
      blockShifting: true,
      edgeMinimization: true,
      parentCentralization: false,
    }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 150,
    zoomView: true,
    dragView: true,
    navigationButtons: false,
  }},
  groups: {json.dumps(groups_json)},
}};

const network = new vis.Network(container, networkData, options);

// ── State ────────────────────────────────────────────────────────────
let showLabels = true;
let physicsOn = true;
const levelShown = {{}};
Object.keys(levelMap).forEach(l => levelShown[l] = true);

// ── Click → info panel ───────────────────────────────────────────────
network.on("click", function(params) {{
  const panel = document.getElementById('info-panel');
  if (params.nodes.length > 0) {{
    const nid = params.nodes[0];
    const node = nodes.get(nid);
    const lvlColors = {json.dumps({k: v["bg"] for k, v in level_colors.items()})};
    const bg = lvlColors[node.cc_level] || '#475569';
    let html = '<div class="meta">' +
      '<span class="level-tag" style="background:' + bg + ';color:#fff">' + node.cc_level + '</span>' +
      '<span class="level-tag" style="background:#334155;color:#94a3b8">' + node.type + '</span>' +
      '</div>';
    html += '<b style="color:#f1f5f9">' + (node.label || node.name) + '</b>';
    html += '<div class="body">' + (node.title || '') + '</div>';
    const connected = network.getConnectedNodes(nid);
    html += '<div class="meta" style="margin-top:6px">Connections: ' + connected.length + '</div>';
    if (connected.length > 0 && connected.length <= 30) {{
      html += '<div class="body">';
      connected.forEach(cid => {{
        const cn = nodes.get(cid);
        html += '<span class="level-tag" style="background:#1e293b;color:#60a5fa;margin:2px">' + (cn.label || cn.name) + '</span>';
      }});
      html += '</div>';
    }}
    document.getElementById('info-content').innerHTML = html;
    panel.style.display = 'block';
  }} else {{
    panel.style.display = 'none';
  }}
}});

// ── Hover → tooltip ──────────────────────────────────────────────────
network.on("hoverNode", function(params) {{
  const tooltip = document.getElementById('tooltip');
  if (params.node !== undefined) {{
    const node = nodes.get(params.node);
    tooltip.innerHTML = '<b>' + (node.label || node.name) + '</b><br><span style="color:#94a3b8;font-size:11px">' + node.cc_level + ' · ' + node.type + '</span>';
    tooltip.style.display = 'block';
  }} else {{
    tooltip.style.display = 'none';
  }}
}});

// ── Search ───────────────────────────────────────────────────────────
document.getElementById('search-box').addEventListener('input', function(e) {{
  const query = e.target.value.toLowerCase().trim();
  if (!query) {{
    nodes.forEach(n => nodes.update({{ id: n.id, borderWidth: 1, size: null, borderWidthSelected: 3 }}));
    return;
  }}
  const updates = [];
  const matches = new Set();
  nodes.forEach(n => {{
    const text = ((n.name||'') + ' ' + (n.label||'') + ' ' + (n.title||'') + ' ' + (n.type||'') + ' ' + (n.cc_level||'')).toLowerCase();
    const match = text.includes(query);
    if (match) matches.add(n.id);
    updates.push({{
      id: n.id,
      borderWidth: match ? 3 : 0.5,
      size: match ? (n.level === 'W1_problem' ? 32 : n.level === 'W2_direction' ? 22 : n.level === 'W4_implementation' ? 18 : 12) : null,
    }});
  }});
  nodes.update(updates);
  if (matches.size === 1) {{
    network.focus(matches.values().next().value, {{ scale: 1.5, animation: true }});
  }}
}});

// ── Toggle level visibility ──────────────────────────────────────────
function toggleLevel(level) {{
  const btn = document.getElementById('btn-' + level);
  levelShown[level] = !levelShown[level];
  const updates = [];
  (levelMap[level] || []).forEach(id => updates.push({{ id, hidden: !levelShown[level] }}));
  nodes.update(updates);
  if (levelShown[level]) btn.classList.remove('active'); else btn.classList.add('active');
}}

// ── Toggle labels ────────────────────────────────────────────────────
function toggleLabels() {{
  showLabels = !showLabels;
  const updates = [];
  nodes.forEach(n => updates.push({{ id: n.id, font: showLabels ? null : {{ size: 0 }} }}));
  nodes.update(updates);
  document.getElementById('btn-labels').textContent = showLabels ? 'Hide Labels' : 'Show Labels';
}}

// ── Toggle physics ───────────────────────────────────────────────────
function togglePhysics() {{
  physicsOn = !physicsOn;
  network.setOptions({{ physics: {{ enabled: physicsOn, solver: 'hierarchicalRepulsion' }} }});
  document.querySelector('[onclick="togglePhysics()"]').textContent = physicsOn ? 'Freeze' : 'Unfreeze';
}}

// ── Fit view ─────────────────────────────────────────────────────────
function fitGraph() {{
  network.fit({{ animation: {{ duration: 800, easingFunction: 'easeInOutQuad' }} }});
}}

// After stabilization, fit to view
network.once("stabilizationIterationsDone", function() {{
  network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
}});
</script>
</body>
</html>"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nGraph saved to: {OUTPUT_PATH}")
    return OUTPUT_PATH


def quick_sample():
    """Populate store with hand-crafted sample data (no LLM needed).

    Models a realistic W2→W3→W4→W5 pyramid for a MARL credit assignment paper.
    """
    import time
    import numpy as np
    from ccchain.config import Config
    from ccchain.core.store import CCStore
    from ccchain.core.ontology import Atom, Edge, Rho

    config = Config()
    # Fresh start
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    # Also clean pickle
    pickle_path = os.path.join(config.graph_dir, "cc_graph.pickle")
    graphml_path = os.path.join(config.graph_dir, "cc_graph.graphml")
    for p in [pickle_path, graphml_path]:
        if os.path.exists(p):
            os.remove(p)

    store = CCStore(config.db_path, config.graph_dir)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rng = np.random.RandomState(42)

    def _emb():
        return rng.randn(1024).astype(np.float32)

    pdf = "sample_paper.pdf"

    # ==== Paper 1: Sinkhorn OT credit assignment ====
    atoms_p1 = [
        # W2 — root problem
        Atom(node_id="P1_W2_credit", name="Credit Assignment in CTDE",
             type="bottleneck", level="W1_problem",
             context="Sparse team rewards cause high variance in policy gradients for individual agents in CTDE-MARL.",
             source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        # W3 — two competing directions
        Atom(node_id="P1_W3_ot", name="Optimal Transport Credit Assignment",
             type="method", level="W2_direction",
             context="Use optimal transport theory to match agent contributions to team rewards with minimal cost.",
             source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P1_W3_shapley", name="Shapley Value Credit Assignment",
             type="method", level="W2_direction",
             context="Use Shapley values from cooperative game theory for axiomatic credit decomposition.",
             source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        # W4 — concrete methods
        Atom(node_id="P1_W4_sinkhorn", name="Sinkhorn OT Credit (λ=0.1)",
             type="method", level="W4_implementation",
             context="Sinkhorn distance with entropy regularization λ=0.1 for fast OT-based credit assignment at scale.",
             source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P1_W4_wasserstein", name="Wasserstein-1 Credit Baseline",
             type="method", level="W4_implementation",
             context="Wasserstein-1 distance as a baseline OT variant without entropy regularization.",
             source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P1_W4_sv", name="Permutation-Sampled Shapley Value",
             type="method", level="W4_implementation",
             context="Monte Carlo permutation sampling to approximate Shapley values for large agent teams.",
             source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        # W5 — code implementations
        Atom(node_id="P1_W5_sk1", name="sinkhorn_credit(cost, reg=0.1)",
             type="component", level="W5_code",
             context="ot.sinkhorn2(cost_matrix, reg=0.1) — core Sinkhorn loop for agent-team reward matching.",
             code_ref="sinkhorn_credit", source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P1_W5_sk2", name="entropy_regularizer(lambda)",
             type="component", level="W5_code",
             context="entropy_regularizer(lambda) — entropy term for Sinkhorn convergence speed vs accuracy trade-off.",
             code_ref="entropy_regularizer", source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P1_W5_ws1", name="wasserstein_credit(cost)",
             type="component", level="W5_code",
             context="ot.emd2(cost_matrix) — exact Wasserstein distance for small-scale credit experiments.",
             code_ref="wasserstein_credit", source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P1_W5_sv1", name="permutation_shapley(n_agents, samples)",
             type="component", level="W5_code",
             context="permutation_shapley(n_agents, n_samples=1000) — MC estimator of Shapley contributions.",
             code_ref="permutation_shapley", source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P1_W5_eval", name="evaluate_on_smac(scenario, method)",
             type="component", level="W5_code",
             context="evaluate_on_smac(scenario='3m', method='sinkhorn') — SMAC benchmark harness.",
             code_ref="evaluate_on_smac", source_pdf=pdf, created_at=now, updated_at=now, embedding=_emb()),
    ]

    edges_p1 = [
        # Downward decomposition (W2→W3→W4→W5)
        Edge(src="P1_W2_credit", relation="decomposes_into", tgt="P1_W3_ot"),
        Edge(src="P1_W2_credit", relation="decomposes_into", tgt="P1_W3_shapley"),
        Edge(src="P1_W3_ot", relation="decomposes_into", tgt="P1_W4_sinkhorn"),
        Edge(src="P1_W3_ot", relation="decomposes_into", tgt="P1_W4_wasserstein"),
        Edge(src="P1_W3_shapley", relation="decomposes_into", tgt="P1_W4_sv"),
        Edge(src="P1_W4_sinkhorn", relation="decomposes_into", tgt="P1_W5_sk1"),
        Edge(src="P1_W4_sinkhorn", relation="decomposes_into", tgt="P1_W5_sk2"),
        Edge(src="P1_W4_wasserstein", relation="decomposes_into", tgt="P1_W5_ws1"),
        Edge(src="P1_W4_sv", relation="decomposes_into", tgt="P1_W5_sv1"),
        Edge(src="P1_W4_sinkhorn", relation="decomposes_into", tgt="P1_W5_eval"),
        # Upward aggregation
        Edge(src="P1_W5_sk1", relation="aggregates_to", tgt="P1_W4_sinkhorn"),
        Edge(src="P1_W5_sk2", relation="aggregates_to", tgt="P1_W4_sinkhorn"),
        Edge(src="P1_W5_ws1", relation="aggregates_to", tgt="P1_W4_wasserstein"),
        Edge(src="P1_W5_sv1", relation="aggregates_to", tgt="P1_W4_sv"),
        Edge(src="P1_W4_sinkhorn", relation="aggregates_to", tgt="P1_W3_ot"),
        Edge(src="P1_W4_wasserstein", relation="aggregates_to", tgt="P1_W3_ot"),
        Edge(src="P1_W4_sv", relation="aggregates_to", tgt="P1_W3_shapley"),
        Edge(src="P1_W3_ot", relation="aggregates_to", tgt="P1_W2_credit"),
        Edge(src="P1_W3_shapley", relation="aggregates_to", tgt="P1_W2_credit"),
        # Cross-direction comparison + improvement
        Edge(src="P1_W3_ot", relation="compares", tgt="P1_W3_shapley",
             rho=Rho(bottleneck="credit_assignment", mechanism="Sinkhorn vs permutation MC",
                     tradeoff="speed vs axiomatic fairness", confidence=0.85)),
        Edge(src="P1_W4_sinkhorn", relation="improves", tgt="P1_W4_wasserstein",
             rho=Rho(bottleneck="credit_assignment", mechanism="entropy regularization stabilizes OT",
                     tradeoff="accuracy vs regularization bias", confidence=0.80)),
    ]

    # ==== Paper 2: QMIX/VDN baseline paper (connected via compares) ====
    atoms_p2 = [
        Atom(node_id="P2_W2_value", name="Value Decomposition in MARL",
             type="bottleneck", level="W1_problem",
             context="IGM (Individual-Global-Max) constraint limits representational capacity in value-based MARL.",
             source_pdf="baseline_qmix.pdf", created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P2_W3_vdn", name="Value Decomposition Networks",
             type="method", level="W2_direction",
             context="VDN: sum-decomposition Q_tot = Σ Q_i, satisfies IGM but limits joint action representation.",
             source_pdf="baseline_qmix.pdf", created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P2_W3_qmix", name="QMIX Monotonic Mixing",
             type="method", level="W2_direction",
             context="QMIX: monotonic mixing network ∂Q_tot/∂Q_i ≥ 0, richer than VDN, still IGM-constrained.",
             source_pdf="baseline_qmix.pdf", created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P2_W4_vdn_impl", name="VDN Sum Decomposition",
             type="method", level="W4_implementation",
             context="Q_tot = sum(Q_i(o_i, a_i)); agent networks share parameters for scalability.",
             source_pdf="baseline_qmix.pdf", created_at=now, updated_at=now, embedding=_emb()),
        Atom(node_id="P2_W4_qmix_impl", name="QMIX Hypernetwork Mixer",
             type="method", level="W4_implementation",
             context="Hypernetwork generates mixing weights |W| = abs(W_raw), ensuring monotonicity.",
             source_pdf="baseline_qmix.pdf", created_at=now, updated_at=now, embedding=_emb()),
    ]

    edges_p2 = [
        Edge(src="P2_W2_value", relation="decomposes_into", tgt="P2_W3_vdn"),
        Edge(src="P2_W2_value", relation="decomposes_into", tgt="P2_W3_qmix"),
        Edge(src="P2_W3_vdn", relation="decomposes_into", tgt="P2_W4_vdn_impl"),
        Edge(src="P2_W3_qmix", relation="decomposes_into", tgt="P2_W4_qmix_impl"),
        Edge(src="P2_W4_vdn_impl", relation="aggregates_to", tgt="P2_W3_vdn"),
        Edge(src="P2_W4_qmix_impl", relation="aggregates_to", tgt="P2_W3_qmix"),
        Edge(src="P2_W3_vdn", relation="aggregates_to", tgt="P2_W2_value"),
        Edge(src="P2_W3_qmix", relation="aggregates_to", tgt="P2_W2_value"),
        # Cross-paper: OT approach replaces QMIX for credit assignment
        Edge(src="P1_W3_ot", relation="replaces", tgt="P2_W3_qmix",
             rho=Rho(bottleneck="credit_assignment", mechanism="OT replaces value decomposition for per-agent credit",
                     tradeoff="no IGM constraint vs extra OT compute cost", confidence=0.75)),
        Edge(src="P1_W4_sinkhorn", relation="extends", tgt="P2_W4_qmix_impl",
             rho=Rho(bottleneck="credit_assignment", mechanism="OT credit can augment QMIX mixing",
                     tradeoff="hybrid credit + value decomposition vs single method", confidence=0.65)),
    ]

    all_atoms = atoms_p1 + atoms_p2
    all_edges = edges_p1 + edges_p2

    result = store.insert_blueprint(all_atoms, all_edges, source_pdf=pdf)
    print(f"  Inserted: {result}")

    store.persist()
    store.db.close()
    print("  Sample data written (2 papers, {} atoms, {} edges)".format(
        len(all_atoms), len(all_edges)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ccchain graph HTML")
    parser.add_argument("--sample", action="store_true",
                        help="Populate with sample data first (requires Ollama + qwen3 + bge-m3)")
    args = parser.parse_args()

    if args.sample:
        quick_sample()

    path = generate_html()
    if path:
        print(f"\nOpen in browser: file:///{path.replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
