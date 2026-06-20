"""cc.py — build a Claim Chain (ccchain) from WM analysis, validate, store, render.

Maps extract.py dicts → ccchain Atom/Edge/Rho. Programmatic (no LLM).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import config as C

# import ccchain (sibling repo) via sys.path
if str(C.FLUXTURBO_DIR) not in sys.path:
    sys.path.insert(0, str(C.FLUXTURBO_DIR))
from ccchain.core.ontology import Atom, Edge, Rho            # noqa: E402
from ccchain.core.gatekeeper import validate                 # noqa: E402
from ccchain.core.store import CCStore                       # noqa: E402
from ccchain.visualize import build_audit_html               # noqa: E402

# valid status palette (build_audit_html toggle buttons index these — KeyError otherwise)
ST_OK = {"verified", "active", "needs_review", "low_reliability", "low_confidence",
         "skipped", "demoted"}


def _atom(nid, name, type_, level, context="", status="active", provenance=None):
    return Atom(node_id=nid, name=name, type=type_, level=level,
                context=context, status=status if status in ST_OK else "active",
                source_pdf="lewm-reacher", provenance=provenance)


def build_cc(analysis: dict, wm_label: str = "LeWM-reacher") -> tuple[list[Atom], list[Edge]]:
    """Construct atoms+edges from extract.probe_all / pca_components / linear_dynamics output."""
    probes = analysis["probes"]
    pca = analysis.get("pca", {})
    dyn = analysis.get("dynamics", {})
    atoms, edges = [], []

    # ── anchor: the WM as a method (target of uses_component / boundary_of) ─
    m_id = f"method:{wm_label}"
    atoms.append(_atom(m_id, f"{wm_label} latent encoder", "method", "W3_approach",
                       context="JEPA world model: ViT-Tiny encoder (192-d) + ARPredictor. "
                               "Latent z=encode(pixels) is the physical-perception layer.",
                       status="verified",
                       provenance={"arch": "ViT-Tiny+ARPredictor", "params_M": 18.0}))

    # ── hypothesis ────────────────────────────────────────────────────────
    h_id = "hyp:latent_encodes_state"
    atoms.append(_atom(h_id, "reacher latent linearly encodes kinematic state",
                       "hypothesis", "W3_approach",
                       context="Probing hypothesis: z linearly decodable into qpos/qvel/obs.",
                       status="needs_review"))
    edges.append(Edge(src=m_id, relation="related_to", tgt=h_id))   # method ↔ hypothesis (wildcard)

    # ── numerical facts (high R²) + boundary concepts (low R²) ────────────
    for p in probes:
        r2 = p["r2"]
        if r2 != r2:   # NaN
            continue
        if r2 >= C.R2_HIGH:
            nid = f"num:r2:{p['name']}"
            atoms.append(_atom(nid, f"R²(z→{p['name']})={r2:.2f}", "numerical", "W4_implementation",
                               context=f"Linear probe: latent z linearly encodes {p['name']} "
                                      f"(test R²={r2:.3f}, control R²={p['ctrl_r2']:.3f}, "
                                      f"selectivity={p['selectivity']:.3f}, n={p['n']}).",
                               status="verified",
                               provenance={"score": round(r2, 4), "metric": "test_R2",
                                           "target": p["name"], "control_R2": round(p["ctrl_r2"], 4)}))
            edges.append(Edge(src=nid, relation="related_to", tgt=h_id))
        elif r2 < C.R2_LOW:
            cid = f"concept:boundary:{p['name']}"
            atoms.append(_atom(cid, f"encoding boundary: {p['name']}", "concept", "W3_approach",
                               context=f"Latent FAILS to linearly encode {p['name']} "
                                      f"(R²={r2:.3f}) — representational limitation.",
                               status="low_reliability",
                               provenance={"score": round(r2, 4), "metric": "test_R2", "target": p["name"]}))
            edges.append(Edge(src=cid, relation="boundary_of", tgt=m_id))  # concept→method (allowed)

    # ── bottleneck: worst-encoded quantity (if any low-R²) ────────────────
    low = [p for p in probes if p["r2"] == p["r2"] and p["r2"] < C.R2_LOW]
    if low:
        worst = min(low, key=lambda p: p["r2"])
        b_id = "bottleneck:representational_limitation"
        atoms.append(_atom(b_id, f"representational_limitation: {worst['name']}",
                           "bottleneck", "W2_direction",
                           context=f"Latent cannot encode {worst['name']} (R²={worst['r2']:.3f}). "
                                   "Candidate for CC→WM compile-back (auxiliary decode head).",
                           status="low_reliability",
                           provenance={"worst_target": worst["name"], "r2": round(worst["r2"], 4)}))

        # partial-observability finding: does 2-frame temporal context resolve the worst bottleneck?
        tp_map = {p["name"]: p["r2"] for p in analysis.get("temporal_probes", [])
                  if p["r2"] == p["r2"]}
        tr2 = tp_map.get(worst["name"])
        if tr2 is not None and tr2 > worst["r2"] + 0.2:
            fid = "finding:temporal_resolves_velocity"
            atoms.append(_atom(
                fid, f"temporal context resolves {worst['name']} (R² {worst['r2']:.2f}→{tr2:.2f})",
                "concept", "W3_approach",
                context=f"Single-frame latent cannot encode {worst['name']} (R²={worst['r2']:.2f}), "
                        f"but a 2-frame context [z(t-1),z(t)] can (R²={tr2:.2f}) → partial-observability "
                        "bottleneck: needs a temporal/memory mechanism, not more capacity.",
                status="verified",
                provenance={"single_frame_R2": round(worst["r2"], 3),
                            "temporal_2frame_R2": round(tr2, 3)}))
            edges.append(Edge(src=fid, relation="related_to", tgt=b_id))

    # ── components (PCA subspaces) ────────────────────────────────────────
    for i, var in enumerate(pca.get("explained_variance", [])[:4]):
        c_id = f"component:pca{i}"
        atoms.append(_atom(c_id, f"latent subspace PC{i+1} (var={var:.2f})", "component",
                           "W5_code", context=f"PCA component {i+1}, explained variance {var:.3f}.",
                           status="active"))
        edges.append(Edge(src=m_id, relation="uses_component", tgt=c_id))  # method→component

    # ── mechanism (linear dynamics fit) ───────────────────────────────────
    if dyn.get("A_shape"):
        mech_id = "method:linear_dynamics"
        atoms.append(_atom(mech_id, "linear latent dynamics ż≈Az", "method", "W4_implementation",
                           context=f"Latent-SINDy linear fit on consecutive z pairs: "
                                  f"one-step R²={dyn.get('r2_linear', 0):.3f}, "
                                  f"top |eig|={[round(abs(e),3) for e in dyn.get('eig_top', [])[:3]]}.",
                           status="verified" if dyn.get("r2_linear", 0) > 0.3 else "low_confidence",
                           provenance={"fit": "lstsq", "r2_linear": round(dyn.get("r2_linear", 0), 4)}))

    # ── verification ──────────────────────────────────────────────────────
    v_id = "verification:probing_protocol"
    atoms.append(_atom(v_id, "probing protocol (Ridge + selectivity)", "verification",
                       "W4_implementation",
                       context="80/20 Ridge linear probe + shuffled-target control task; "
                               "R² reported on held-out test.",
                       status="verified",
                       provenance={"protocol": "Ridge+control_task"}))
    edges.append(Edge(src=v_id, relation="validates", tgt=h_id))   # verification→hypothesis (allowed)

    return atoms, edges


def render_cc(atoms, edges, out_html: Path, out_json: Path, label="WM→CC",
              db_path: Path | None = None, graph_dir: Path | None = None):
    db_path = db_path or (C.OUTPUT_DIR / "cc.db")
    graph_dir = graph_dir or (C.OUTPUT_DIR / "cc_graph")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    errs = validate(atoms, edges)
    print(f"[cc] validate → {len(errs)} errors")
    for e in errs[:8]:
        print("   ", e)

    store = CCStore(str(db_path), str(graph_dir))
    na = store.upsert_atoms(atoms)
    ne = store.upsert_edges(edges)
    try:
        store.persist()
    except Exception as ex:
        print(f"[cc] persist note: {ex}")
    out_html = Path(out_html)
    build_audit_html(store, reports=[], output_path=str(out_html), title=f"ccchain · {label}")
    Path(out_json).write_text(json.dumps(
        {"atoms": [a.to_dict() for a in atoms], "edges": [e.to_dict() for e in edges]},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[cc] {na} atoms upserted, {ne} edges; HTML → {out_html}; JSON → {out_json}")
    return store
