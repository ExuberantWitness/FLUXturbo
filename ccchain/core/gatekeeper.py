"""5-layer validation rules — pure functions, no LLM dependency."""

from __future__ import annotations

from ccchain.core.ontology import (
    ATOM_TYPE_SET,
    CC_EDGE_TYPES,
    HIERARCHY_EDGES,
    LEVEL_ORDER,
    STRONG_CAUSAL_EDGES,
    TYPE_COMPATIBILITY,
    Atom,
    Edge,
)


def validate(atoms: list[Atom], edges: list[Edge]) -> list[dict]:
    """Run all 5 validation layers. Returns list of error dicts (empty = pass).

    Rule 1 — Schema: type/level/relation in legal sets, edge endpoints exist.
    Rule 2 — Type Compatibility: edge endpoints in TYPE_COMPATIBILITY matrix.
    Rule 3 — Rho Completeness: strong-causal edges must carry rho evidence.
    Rule 4 — Level Consistency: hierarchy edges must cross levels correctly.
    Rule 5 — Dedup Detection: same name + type + level → duplicate suggestion.
    """
    errors: list[dict] = []
    atom_ids: set[str] = {a.node_id for a in atoms}

    for atom in atoms:
        # Rule 1 — Schema
        if not atom.node_id:
            errors.append(_err("R1", atom.node_id, "Atom node_id is empty"))
        if atom.type not in ATOM_TYPE_SET:
            errors.append(_err("R1", atom.node_id, f"Invalid atom type: {atom.type!r}"))
        if atom.level not in LEVEL_ORDER:
            errors.append(_err("R1", atom.node_id, f"Invalid level: {atom.level!r}"))

    all_edge_types = set(CC_EDGE_TYPES) | set(HIERARCHY_EDGES)

    for i, edge in enumerate(edges):
        eid = f"edge[{i}] {edge.src}→{edge.tgt}"

        # Rule 1 — Schema
        if edge.relation not in all_edge_types:
            errors.append(_err("R1", eid, f"Invalid relation: {edge.relation!r}"))
        if edge.src not in atom_ids:
            errors.append(_err("R1", eid, f"Source {edge.src!r} not in atoms"))
        if edge.tgt not in atom_ids:
            errors.append(_err("R1", eid, f"Target {edge.tgt!r} not in atoms"))

        # Rule 3 — Rho Completeness
        if edge.relation in STRONG_CAUSAL_EDGES and edge.rho is None:
            errors.append(_err("R3", eid, "Strong-causal edge requires rho evidence"))

        # Rule 2 — Type Compatibility
        src_atom = _find(edge.src, atoms)
        tgt_atom = _find(edge.tgt, atoms)
        if src_atom and tgt_atom:
            pair = (src_atom.type, tgt_atom.type)
            compat = TYPE_COMPATIBILITY.get(edge.relation, set())
            if edge.relation not in ("related_to", "background") and compat and pair not in compat:
                errors.append(_err("R2", eid, f"Incompatible: {pair[0]}→{pair[1]}"))

        # Rule 4 — Level Consistency
        if src_atom and tgt_atom and edge.relation in set(HIERARCHY_EDGES):
            src_ord = LEVEL_ORDER.get(src_atom.level)
            tgt_ord = LEVEL_ORDER.get(tgt_atom.level)
            if src_ord is not None and tgt_ord is not None:
                if edge.relation == "aggregates_to" and not (src_ord > tgt_ord):
                    errors.append(_err("R4", eid,
                        f"AGGREGATES_TO must go upward (W{src_ord+2}→W{tgt_ord+2})"))
                if edge.relation == "decomposes_into" and not (src_ord < tgt_ord):
                    errors.append(_err("R4", eid,
                        f"DECOMPOSES_INTO must go downward (W{src_ord+2}→W{tgt_ord+2})"))

    # Rule 5 — Dedup Detection
    seen: dict[tuple[str, str, str], str] = {}
    for atom in atoms:
        key = (atom.name.lower(), atom.type, atom.level)
        if key in seen:
            errors.append(_err("R5", atom.node_id,
                f"Duplicate of {seen[key]}: name={atom.name!r}, type={atom.type}, level={atom.level}"))
        else:
            seen[key] = atom.node_id

    return errors


def _err(rule: str, target: str, message: str) -> dict:
    return {"rule": rule, "target": target, "message": message}


def _find(node_id: str, atoms: list[Atom]) -> Atom | None:
    for a in atoms:
        if a.node_id == node_id:
            return a
    return None
