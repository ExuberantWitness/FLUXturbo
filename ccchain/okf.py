"""Open Knowledge Format (OKF) v0.1 interop — bidirectional portability.

OKF (https://google.github.io/open-knowledge-format/) is a vendor-neutral,
agent/human-friendly standard: a *bundle* is a directory of markdown files
where each file is one *concept*, the file path is its identity, a small YAML
frontmatter block carries structured fields (`type` required; title,
description, resource, tags, timestamp optional), and normal markdown links
turn the directory into a graph. Optional `index.md` (progressive disclosure)
and `log.md` (change history) are reserved filenames.

This module makes ccchain's atom/edge graph portable as OKF:

    from ccchain.okf import export_okf, import_okf

    export_okf(store, "my_bundle/")        # ccchain graph  -> OKF bundle
    import_okf("my_bundle/", store)        # OKF bundle     -> ccchain graph

Every exported concept carries the required `type` field plus ccchain's richer
state (level, status, provenance, CoE verdicts) under `ccchain_*` frontmatter
keys — OKF is minimally opinionated and allows arbitrary extra fields. Edges
become a `## Relations` section of typed markdown links, so the bundle is a
navigable graph in any markdown viewer (GitHub, Obsidian) with no ccchain
dependency.
"""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

from ccchain.core.ontology import (
    ATOM_TYPE_SET,
    LEVEL_DEFAULT_TYPE,
    LEVELS,
    LEVEL_ORDER,
    Atom,
    Edge,
)

_RESERVED_FILES = {"index.md", "log.md"}
_RELATION_LINE = re.compile(r"^-\s+([a-z_]+)\s*→?\s*\[([^\]]*)\]\(([^)]+)\)\s*$")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


# ---------------------------------------------------------------------------
# Path / slug helpers (file path = concept identity)
# ---------------------------------------------------------------------------
def _slug(text: str) -> str:
    """Filesystem-safe slug from a node_id or name."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip()).strip("._")
    return (s or "concept")[:80]


def _concept_path(level: str, node_id: str) -> str:
    """Relative concept path: <level>/<slug>.md (identity within a bundle)."""
    return f"{level}/{_slug(node_id)}.md"


# ---------------------------------------------------------------------------
# Export: ccchain graph → OKF bundle
# ---------------------------------------------------------------------------
def export_okf(
    store,
    bundle_dir: str,
    *,
    source_pdf: str | None = None,
    include_log: bool = True,
) -> str:
    """Export the store's atoms + edges to an OKF v0.1 bundle.

    Args:
        store: a :class:`~ccchain.core.store.CCStore`.
        bundle_dir: destination directory (created if missing).
        source_pdf: if set, export only atoms whose source_pdf matches.
        include_log: write a log.md change history.

    Returns:
        Absolute path of the bundle directory.
    """
    atoms = _gather_atoms(store, source_pdf)
    edges = _gather_edges(store)

    os.makedirs(bundle_dir, exist_ok=True)
    # node_id → relative concept path, for cross-links.
    path_of: dict[str, str] = {a.node_id: _concept_path(a.level, a.node_id) for a in atoms}
    title_of: dict[str, str] = {a.node_id: a.name for a in atoms}

    for atom in atoms:
        _write_concept(bundle_dir, atom, edges, path_of, title_of)

    _write_index(bundle_dir, atoms, path_of)
    if include_log:
        _write_log(bundle_dir, atoms)

    return os.path.abspath(bundle_dir)


def _gather_atoms(store, source_pdf: str | None) -> list[Atom]:
    out: list[Atom] = []
    for lvl in LEVELS:
        for a in store.query_by_level(lvl, status=None):
            if source_pdf and a.source_pdf != source_pdf:
                continue
            out.append(a)
    return out


def _gather_edges(store) -> list[Edge]:
    rows = store.db.execute(
        "SELECT src, tgt, relation, weight, rho_json, provenance FROM cc_edges"
    ).fetchall()
    import json
    edges: list[Edge] = []
    for r in rows:
        rho = None
        if r["rho_json"]:
            try:
                from ccchain.core.ontology import Rho
                rho = Rho.from_dict(json.loads(r["rho_json"]))
            except Exception:
                rho = None
        prov = json.loads(r["provenance"]) if r["provenance"] else None
        edges.append(Edge(
            src=r["src"], tgt=r["tgt"], relation=r["relation"],
            weight=r["weight"] or 1.0, rho=rho, provenance=prov,
        ))
    return edges


def _write_concept(bundle_dir, atom, edges, path_of, title_of):
    rel_path = path_of[atom.node_id]
    full = os.path.join(bundle_dir, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)

    # outgoing + incoming edges (typed links)
    out_links = [(e.relation, e.tgt) for e in edges if e.src == atom.node_id]
    in_links = [(e.relation, e.src) for e in edges if e.tgt == atom.node_id]

    frontmatter = {
        # OKF standard fields (type required; rest optional)
        "type": atom.type,
        "title": atom.name,
        "description": (atom.context or "")[:500],
        "resource": atom.source_pdf or "",
        "tags": list(atom.tags) if atom.tags else [],
        "timestamp": atom.updated_at or "",
        # ccchain extensions (OKF allows arbitrary extra frontmatter)
        "ccchain_node_id": atom.node_id,
        "ccchain_level": atom.level,
        "ccchain_status": atom.status,
        "ccchain_provenance": atom.provenance or {},
    }
    if atom.source_refs:
        frontmatter["ccchain_source_refs"] = list(atom.source_refs)
    if atom.code_ref:
        frontmatter["ccchain_code_ref"] = atom.code_ref

    body_lines: list[str] = []
    body_lines.append(atom.context or "")
    body_lines.append("")

    if atom.code_body:
        body_lines.append("## Code")
        body_lines.append("```python")
        body_lines.append(atom.code_body)
        body_lines.append("```")
        body_lines.append("")

    if out_links or in_links:
        body_lines.append("## Relations")
        for rel, tgt in out_links:
            tgt_path = path_of.get(tgt)
            if tgt_path:
                body_lines.append(f"- {rel} → [{title_of.get(tgt, tgt)}]({tgt_path})")
        for rel, src in in_links:
            src_path = path_of.get(src)
            if src_path:
                body_lines.append(f"- {rel} ← [{title_of.get(src, src)}]({src_path})")
        body_lines.append("")

    if atom.provenance:
        body_lines.append("## Provenance")
        body_lines.append("```json")
        import json as _json
        body_lines.append(_json.dumps(atom.provenance, indent=2, ensure_ascii=False))
        body_lines.append("```")
        body_lines.append("")

    md = "---\n" + yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, width=1000
    ) + "---\n\n" + "\n".join(body_lines).rstrip() + "\n"
    with open(full, "w", encoding="utf-8") as f:
        f.write(md)


def _write_index(bundle_dir, atoms, path_of):
    """index.md — progressive disclosure: the W1→W5 hierarchy at a glance."""
    lines = ["# Index", "", "ccchain knowledge pyramid (W1→W5).", ""]
    for lvl in LEVELS:
        lvl_atoms = [a for a in atoms if a.level == lvl]
        lines.append(f"## {lvl}  ({len(lvl_atoms)})")
        if not lvl_atoms:
            lines.append("_(none)_")
        for a in sorted(lvl_atoms, key=lambda x: x.name):
            lines.append(f"- [{a.name}]({path_of[a.node_id]})  · `{a.type}` · {a.status}")
        lines.append("")
    with open(os.path.join(bundle_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _write_log(bundle_dir, atoms):
    """log.md — chronological change history."""
    lines = ["# Log", "", "Change history by atom updated_at.", ""]
    for a in sorted(atoms, key=lambda x: x.updated_at or ""):
        lines.append(f"- {a.updated_at or '?'} · `{a.node_id}` · {a.name} · {a.status}")
    with open(os.path.join(bundle_dir, "log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Import: OKF bundle → ccchain graph
# ---------------------------------------------------------------------------
def import_okf(bundle_dir: str, store) -> dict:
    """Parse an OKF bundle into atoms + edges and insert into the store.

    Reads every `.md` concept (skipping reserved `index.md`/`log.md`), maps
    frontmatter back to atoms, and reconstructs edges from the `## Relations`
    typed markdown links.

    Returns: {atoms_imported, edges_imported, concepts_skipped}.
    """
    concepts = _read_bundle(bundle_dir)  # list of (abspath, frontmatter, body)

    # path (relative to bundle) → node_id, for link resolution
    path_to_node: dict[str, str] = {}
    parsed: list[tuple[dict, str]] = []  # (frontmatter, body)
    skipped = 0
    for abs_path, fm, body in concepts:
        node_id = _node_id_from(fm, abs_path, bundle_dir)
        rel = os.path.relpath(abs_path, bundle_dir).replace(os.sep, "/")
        path_to_node[rel] = node_id
        parsed.append((fm, body, node_id, rel))

    # First pass: build atoms, collecting a node_id → atom map
    atoms: list[Atom] = []
    node_to_atom: dict[str, Atom] = {}
    for fm, body, node_id, rel in parsed:
        atom = _frontmatter_to_atom(fm, node_id, body)
        if atom is None:
            skipped += 1
            continue
        atoms.append(atom)
        node_to_atom[node_id] = atom

    # Second pass: edges from ## Relations links
    edges: list[Edge] = []
    seen_nodes = set(node_to_atom)
    for fm, body, node_id, rel in parsed:
        if node_id not in seen_nodes:
            continue
        for relation, target_node in _parse_relations(body, path_to_node, bundle_dir):
            if target_node in seen_nodes and target_node != node_id:
                edges.append(Edge(src=node_id, relation=relation, tgt=target_node))

    store.insert_blueprint(atoms, edges, "okf_import")

    return {
        "atoms_imported": len(atoms),
        "edges_imported": len(edges),
        "concepts_skipped": skipped,
    }


def _read_bundle(bundle_dir: str):
    """Walk bundle_dir, yield (abs_path, frontmatter_dict, body) per concept."""
    out = []
    for dp, _dirs, fns in os.walk(bundle_dir):
        for fn in fns:
            if not fn.endswith(".md") or fn.lower() in _RESERVED_FILES:
                continue
            abs_path = os.path.join(dp, fn)
            fm, body = _split_frontmatter(abs_path)
            if fm is None:
                continue
            out.append((abs_path, fm, body))
    return out


def _split_frontmatter(path: str) -> tuple[dict | None, str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---"):
        return {}, text
    parts = text[3:].split("\n---", 1)
    if len(parts) != 2:
        return {}, text
    try:
        fm = yaml.safe_load(parts[0]) or {}
    except yaml.YAMLError:
        return None, ""
    return fm, parts[1].lstrip("\n")


def _node_id_from(fm: dict, abs_path: str, bundle_dir: str) -> str:
    if fm.get("ccchain_node_id"):
        return str(fm["ccchain_node_id"])
    if fm.get("node_id"):
        return str(fm["node_id"])
    # fall back to the concept path (OKF: file path is identity)
    rel = os.path.relpath(abs_path, bundle_dir).replace(os.sep, "/")
    return rel[:-3]  # strip .md


def _frontmatter_to_atom(fm: dict, node_id: str, body: str) -> Atom | None:
    atom_type = str(fm.get("type") or "")
    if atom_type not in ATOM_TYPE_SET:
        # OKF is open — coerce unknown types to a safe default, record original.
        original = atom_type
        atom_type = "concept"
    else:
        original = None

    level = str(fm.get("ccchain_level") or fm.get("level") or "")
    if level not in LEVEL_ORDER:
        # infer level from the body/path is unreliable; default per type-ish.
        level = "W3_approach"

    # description from frontmatter, else first non-empty body line
    context = str(fm.get("description") or "")
    if not context:
        context = body.strip().split("\n", 1)[0][:500]

    prov = fm.get("ccchain_provenance")
    if isinstance(prov, str):
        import json
        try:
            prov = json.loads(prov)
        except Exception:
            prov = {}
    if original:
        prov = dict(prov or {})
        prov["okf_original_type"] = original

    return Atom(
        node_id=node_id,
        name=str(fm.get("title") or node_id),
        type=atom_type,
        level=level,
        context=context,
        source_pdf=str(fm.get("resource") or "") or None,
        tags=list(fm.get("tags")) if fm.get("tags") else None,
        status=str(fm.get("ccchain_status") or "active"),
        provenance=prov,
    )


def _parse_relations(body: str, path_to_node: dict, bundle_dir: str):
    """Yield (relation, target_node_id) from a ## Relations section."""
    in_relations = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_relations = stripped.lower().startswith("## relation")
            continue
        if not in_relations:
            continue
        # two encodings: "- rel → [text](path)" or "- rel ← [text](path)"
        m = _LINK_RE.search(stripped)
        if not m:
            continue
        # relation = token(s) before the first '['
        prefix = stripped[: m.start()].lstrip("- ").strip()
        relation = prefix.replace("→", "").replace("←", "").strip() or "related_to"
        if relation not in _ALL_RELATIONS:
            relation = "related_to"
        target_path = _strip_md_underscores(m.group(2))
        target_node = _resolve_link(target_path, path_to_node, bundle_dir)
        if target_node:
            yield relation, target_node


_ALL_RELATIONS = {
    "extends", "improves", "replaces", "adapts", "uses_component", "compares",
    "background", "implements", "validates", "boundary_of", "related_to",
    "aggregates_to", "decomposes_into",
}


def _strip_md_underscores(p: str) -> str:
    # export wraps paths in _..._ to keep GitHub from rendering them as links
    # inside list items oddly; strip those markers.
    p = p.strip()
    if p.startswith("_") and p.endswith("_"):
        p = p[1:-1]
    return p


def _resolve_link(link: str, path_to_node: dict, bundle_dir: str) -> str | None:
    """Resolve a markdown link target to a node_id."""
    link = link.split("#", 1)[0].split("?")[0]
    if not link:
        return None
    # absolute path within bundle
    rel = link.lstrip("./").lstrip("\\")
    if rel in path_to_node:
        return path_to_node[rel]
    # try normalising
    norm = rel.replace("\\", "/")
    if norm in path_to_node:
        return path_to_node[norm]
    # basename match
    base = os.path.basename(norm)
    for k, v in path_to_node.items():
        if os.path.basename(k) == base:
            return v
    return None
