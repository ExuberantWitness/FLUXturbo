"""6-layer validation rules — pure functions, no LLM dependency.

R1 Schema: type/level/relation in legal sets, edge endpoints exist.
R2 Type Compatibility: edge endpoints in TYPE_COMPATIBILITY matrix.
R3 Rho Completeness: strong-causal edges must carry rho evidence.
R4 Level Consistency: hierarchy edges must cross levels correctly.
R5 Dedup Detection: same name + type + level → duplicate suggestion.
R6 Provenance Presence: numerical/citation/method/solution/experiment must have provenance.

v0.5: R7 (type-level 1:1 consistency) removed — type and level are decoupled.
"""

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


# Type → required provenance keys (R6)
PROVENANCE_REQUIREMENTS: dict[str, list[str]] = {
    "numerical":  ["score"],
    "citation":   ["raw_citation"],
    "method":     [],      # any non-empty provenance
    "solution":   [],
    "experiment": [],
}

# Type demotion map when provenance missing beyond refine rounds (R6 fallback)
# Same-layer demotion only; preserves level, loses CoE eligibility.
TYPE_DEMOTION_MAP: dict[str, str] = {
    "numerical": "conclusion",   # W4 → W4
    "citation":  "concept",      # W3 → W3
    # method/solution/experiment have no same-layer demotion target; marked needs_review
}


def validate(atoms: list[Atom], edges: list[Edge]) -> list[dict]:
    """Run all 6 validation layers. Returns list of error dicts (empty = pass)."""
    errors: list[dict] = []
    atom_ids: set[str] = {a.node_id for a in atoms}

    for atom in atoms:
        # Rule 1 — Schema
        if not atom.node_id:
            errors.append(_err("R1", atom.node_id, "Atom node_id is empty"))
        if atom.type not in ATOM_TYPE_SET:
            errors.append(_err("R1", atom.node_id, f"Invalid atom type: {atom.type!r}"))
            continue  # downstream rules assume valid type
        if atom.level not in LEVEL_ORDER:
            errors.append(_err("R1", atom.node_id, f"Invalid level: {atom.level!r}"))
            continue

        # Rule 6 — Provenance Presence (by type)
        if atom.type in PROVENANCE_REQUIREMENTS:
            required_keys = PROVENANCE_REQUIREMENTS[atom.type]
            if not atom.provenance:
                errors.append(_err("R6", atom.node_id,
                    f"{atom.type!r} atom requires provenance (missing)"))
            elif required_keys:
                missing = [k for k in required_keys if atom.provenance.get(k) is None]
                if missing:
                    errors.append(_err("R6", atom.node_id,
                        f"{atom.type!r} atom provenance missing keys: {missing}"))

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


def apply_r6_demotions(atoms: list[Atom]) -> int:
    """Fallback after refiner exhaustion: mutate atoms in-place to satisfy R6.

    - numerical without provenance.score → type='conclusion', status='demoted'
    - citation without provenance.raw_citation → type='concept', status='demoted'
    - method/solution/experiment without any provenance → status='needs_review' (no type change)

    Returns count of atoms mutated. Caller should re-validate after.
    """
    count = 0
    for atom in atoms:
        if atom.type not in PROVENANCE_REQUIREMENTS:
            continue
        required = PROVENANCE_REQUIREMENTS[atom.type]
        ok = bool(atom.provenance)
        if ok and required:
            ok = all(atom.provenance.get(k) is not None for k in required)
        if ok:
            continue

        # R6 still failing — apply demotion (type changes; level is decoupled, stays)
        if atom.type in TYPE_DEMOTION_MAP:
            new_type = TYPE_DEMOTION_MAP[atom.type]
            atom.type = new_type
            atom.status = "demoted"
            count += 1
        else:
            # method/solution/experiment — no same-layer demotion target
            atom.status = "needs_review"
            count += 1
    return count


def _err(rule: str, target: str, message: str) -> dict:
    return {"rule": rule, "target": target, "message": message}


def _find(node_id: str, atoms: list[Atom]) -> Atom | None:
    for a in atoms:
        if a.node_id == node_id:
            return a
    return None
