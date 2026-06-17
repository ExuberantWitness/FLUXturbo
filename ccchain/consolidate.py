"""Cross-paper hierarchical consolidation utility.

Top-down (W2→W3→W4→W5) parent-grouped parallel merge of semantically
duplicate atoms across papers. Standalone utility — NOT an SDK method and
NOT a plugin ABC. Mirrors the role of ``ccchain.visualize``.

Public entry::

    from ccchain.consolidate import consolidate
    report = consolidate(store, config=config)

Algorithm (per the v0.4 plan): for each level top-down, cluster the level's
LIVE atoms WITHIN each parent-group by embedding cosine similarity, then ask
an LLM merge-arbiter (in parallel across clusters) whether each cluster is
truly one concept. Confirmed merges rewrite a canonical atom and flip the
duplicates to ``status="merged"``, rewiring their edges. Idempotent: merged
atoms are excluded from future passes.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ccchain.config import Config
from ccchain.core.ontology import Atom, Edge, LEVEL_ORDER, TYPE_TO_LEVEL
from ccchain.core.store import CCStore

# Top-down processing order: classify by direction first, subdivide downward.
_LEVELS_TOP_DOWN = [
    "W2_problem_analysis",
    "W3_solution_direction",
    "W4_concrete_solution",
    "W5_code_implementation",
]

# Allowed atom types per level (3 each, mirrors _REDUCE_PROMPT in reduction.py).
_LEVEL_ALLOWED_TYPES: dict[str, list[str]] = {
    "W2_problem_analysis": ["problem", "bottleneck", "hypothesis"],
    "W3_solution_direction": ["method", "citation", "concept"],
    "W4_concrete_solution": ["solution", "numerical", "conclusion"],
    "W5_code_implementation": ["component", "experiment", "verification"],
}

# Atoms in these statuses are skipped (already-merged, or evaluate() temporaries).
_SKIP_STATUSES = frozenset({"merged", "transient"})

_HIERARCHY_RELATIONS = ("aggregates_to", "decomposes_into")


# ---------------------------------------------------------------------------
# Merge-arbiter prompt (new — no equivalence-judging prompt exists elsewhere)
# ---------------------------------------------------------------------------
_MERGE_ARBITER_PROMPT = """\
You are a strict merge arbiter for a cross-paper knowledge graph.

CONTEXT: Two or more atoms at level {level} live UNDER THE SAME parent
({parent_kind}). They were flagged as candidate duplicates by cosine embedding
similarity. Judge whether they are truly the SAME concept across papers and
should be merged into ONE canonical atom, or whether the similarity is
coincidental and they must stay separate.

CANDIDATE ATOMS (JSON):
{atoms_json}

RULES:
- Merge ONLY if they refer to the same research concept/method/problem,
  differing at most in phrasing, granularity, or citation framing.
- Do NOT merge if they are sibling sub-concepts, related-but-distinct methods,
  or same-name-different-meaning (e.g. "attention" as a mechanism vs as an
  evaluation metric).
- If merging: pick ONE canonical name (most general/standard), the best type
  from {allowed_types}, and a context that unifies the candidates.

Return JSON ONLY:
{{
  "merge": true|false,
  "canonical_name": "...",
  "canonical_type": "...",          // one of {allowed_types} if merge=true
  "canonical_context": "...",       // unified description if merge=true
  "reasoning": "one sentence why merge or not"
}}
"""


# ---------------------------------------------------------------------------
# Consolidator (plain class — NOT a plugin ABC)
# ---------------------------------------------------------------------------
class HierarchicalConsolidator:
    """LLM merge-arbiter wrapper. Plain class, no ABC inheritance."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        similarity_threshold: float = 0.85,
        majority_k: int = 3,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.majority_k = majority_k

    def judge_cluster(
        self,
        atoms: list[Atom],
        level: str,
        parent_kind: str,
    ) -> dict | None:
        """Ask the LLM whether `atoms` should merge. Returns arbiter JSON or None."""
        from ccchain.core.llm import chat_json_majority

        atoms_json = [
            {"node_id": a.node_id, "name": a.name, "type": a.type,
             "context": (a.context or "")[:400]}
            for a in atoms
        ]
        prompt = _MERGE_ARBITER_PROMPT.format(
            level=level,
            parent_kind=parent_kind,
            atoms_json=atoms_json,
            allowed_types=_LEVEL_ALLOWED_TYPES[level],
        )
        try:
            return chat_json_majority(
                [{"role": "user", "content": prompt}],
                base_url=self.base_url, api_key=self.api_key,
                model=self.model, k=self.majority_k,
            )
        except Exception:
            return None


@dataclass
class _MergeDecision:
    merge: bool
    canonical_id: str = ""
    canonical_name: str = ""
    canonical_type: str = ""
    canonical_context: str = ""
    reasoning: str = ""
    dup_ids: list[str] = field(default_factory=list)
    parent_id: str | None = None
    level: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """n×n cosine similarity matrix (float32). Identity-sized zero for n<=1."""
    n = len(vectors)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)
    X = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def _candidate_clusters(
    atoms: list[Atom],
    emb_map: dict[str, np.ndarray],
    threshold: float,
) -> list[list[Atom]]:
    """Single-linkage clustering by cosine >= threshold within one parent-group.

    Returns clusters of len>=2 (singletons dropped). Atoms lacking an embedding
    are excluded.
    """
    kept = [a for a in atoms if a.node_id in emb_map]
    if len(kept) < 2:
        return []
    S = _cosine_matrix([emb_map[a.node_id] for a in kept])
    n = len(kept)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= threshold:
                parent[find(i)] = find(j)

    buckets: dict[int, list[Atom]] = defaultdict(list)
    for i, a in enumerate(kept):
        buckets[find(i)].append(a)
    return [c for c in buckets.values() if len(c) >= 2]


def _parent_grouping(
    store: CCStore, level: str, atoms: list[Atom],
) -> dict[str | None, list[Atom]]:
    """Group same-level atoms by their parent one level above.

    Returns {parent_node_id: [atoms]}. Orphans (no parent) land under key None.
    Top level (W2) is a single group keyed by "__root__".
    """
    if level == "W2_problem_analysis":
        return {"__root__": list(atoms)}

    groups: dict[str | None, list[Atom]] = defaultdict(list)
    target_level_idx = LEVEL_ORDER[level] - 1
    for a in atoms:
        parent_id: str | None = None
        seen: set[str] = set()
        for rel in _HIERARCHY_RELATIONS:
            for nid in store.get_neighbors(a.node_id, relation=rel, direction="both"):
                if nid in seen:
                    continue
                seen.add(nid)
                p = store.query_by_id(nid)
                if p is not None and LEVEL_ORDER.get(p.level, -1) == target_level_idx:
                    parent_id = nid
                    break
            if parent_id is not None:
                break
        groups[parent_id].append(a)
    return dict(groups)


def _adjudicate_cluster(
    consolidator: HierarchicalConsolidator,
    cluster: list[Atom],
    level: str,
    parent_id: str | None,
) -> _MergeDecision:
    parent_kind = (
        "no parent (top level W2)" if parent_id in (None, "__root__")
        else f"parent {parent_id}"
    )
    resp = consolidator.judge_cluster(cluster, level, parent_kind)
    if not resp or not resp.get("merge"):
        return _MergeDecision(merge=False, reasoning=str(resp))

    raw_type = resp.get("canonical_type") or ""
    if raw_type not in _LEVEL_ALLOWED_TYPES[level]:
        raw_type = _LEVEL_ALLOWED_TYPES[level][0]

    canonical = cluster[0]  # reuse an existing node_id (mirrors _auto_dedup)
    return _MergeDecision(
        merge=True,
        canonical_id=canonical.node_id,
        canonical_name=resp.get("canonical_name") or canonical.name,
        canonical_type=raw_type,
        canonical_context=resp.get("canonical_context") or canonical.context,
        reasoning=resp.get("reasoning", ""),
        dup_ids=[a.node_id for a in cluster[1:]],
        parent_id=None if parent_id == "__root__" else parent_id,
        level=level,
    )


def _apply_merges(
    store: CCStore, decisions: list[_MergeDecision],
) -> tuple[list[Atom], list[Atom], list[Edge]]:
    """Materialize confirmed merges against the store.

    Returns (canonical_atoms, merged_atoms, new_edges). Caller is responsible
    for re-embedding canonical atoms, deleting dup-touching edges, upserting,
    and rebuilding the graph.
    """
    canonical_atoms: list[Atom] = []
    merged_atoms: list[Atom] = []
    new_edges: list[Edge] = []

    for d in decisions:
        if not d.merge:
            continue
        canon = store.query_by_id(d.canonical_id)
        if canon is None:
            continue

        # 1. Rewrite canonical in place.
        canon.name = d.canonical_name
        canon.type = d.canonical_type
        canon.context = d.canonical_context
        canon.status = "active"  # re-queue for audit (context changed)
        prov = dict(canon.provenance or {})
        prov["phase"] = "consolidate"
        merged_from = list(prov.get("merged_from") or [])
        for did in d.dup_ids:
            if did not in merged_from:
                merged_from.append(did)
        prov["merged_from"] = merged_from
        if d.parent_id:
            prov["parent"] = d.parent_id
        canon.provenance = prov
        canonical_atoms.append(canon)

        # 2. Flip dups to merged.
        for dup_id in d.dup_ids:
            dup = store.query_by_id(dup_id)
            if dup is None:
                continue
            dup.status = "merged"
            dup_prov = dict(dup.provenance or {})
            dup_prov["merged_into"] = d.canonical_id
            dup_prov["phase"] = "consolidate"
            dup.provenance = dup_prov
            merged_atoms.append(dup)

        # 3. Rewire every dup edge to the canonical (skip self-loops).
        for dup_id in d.dup_ids:
            for tgt, rel in store.edge_targets(dup_id):
                if tgt == d.canonical_id or tgt == dup_id:
                    continue
                new_edges.append(Edge(src=d.canonical_id, tgt=tgt, relation=rel))
            for src, rel in store.edge_sources(dup_id):
                if src == d.canonical_id or src == dup_id:
                    continue
                new_edges.append(Edge(src=src, tgt=d.canonical_id, relation=rel))

    return canonical_atoms, merged_atoms, new_edges


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------
def consolidate(
    store: CCStore,
    *,
    config: Config | None = None,
    on_progress: "callable | None" = None,
) -> dict:
    """Top-down parent-grouped parallel merge of cross-paper duplicate atoms.

    Idempotent: atoms already flipped to ``status="merged"`` are skipped on
    subsequent runs. Returns a report dict (atoms_merged, clusters_formed,
    clusters_confirmed, canonical_ids, per_level).
    """
    config = config or Config()
    consolidator = HierarchicalConsolidator(
        base_url=config.llm_base_url, api_key=config.llm_api_key,
        model=config.llm_model,
        similarity_threshold=config.consolidate_similarity_threshold,
        majority_k=config.consolidate_majority_k,
    )

    def _progress(stage: str, pct: float):
        if on_progress:
            on_progress(stage, pct)

    report: dict[str, Any] = {
        "levels_processed": [],
        "atoms_merged": 0,
        "clusters_formed": 0,
        "clusters_confirmed": 0,
        "canonical_ids": [],
        "per_level": {},
    }

    n_levels = len(_LEVELS_TOP_DOWN)
    for li, level in enumerate(_LEVELS_TOP_DOWN):
        _progress("consolidating", li / n_levels)
        atoms = [
            a for a in store.query_by_level(level, status=None)
            if a.status not in _SKIP_STATUSES
        ]
        emb_map = store.query_embeddings_by_level(level)
        groups = _parent_grouping(store, level, atoms)

        # Candidate clusters within each parent-group.
        all_clusters: list[tuple[list[Atom], str | None]] = []
        for parent_id, grp in groups.items():
            if parent_id is None:
                continue  # orphans: no hierarchy context, skip
            for cluster in _candidate_clusters(grp, emb_map, consolidator.similarity_threshold):
                all_clusters.append((cluster, parent_id))
        report["clusters_formed"] += len(all_clusters)

        if not all_clusters:
            report["per_level"][level] = {"candidates": 0, "merged": 0}
            continue

        # Parallel LLM adjudication — judge_cluster is read-only on the store.
        decisions: list[_MergeDecision] = []
        max_workers = max(1, min(8, len(all_clusters)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(_adjudicate_cluster, consolidator, c, level, p)
                for c, p in all_clusters
            ]
            for f in futs:
                decisions.append(f.result())

        confirmed = [d for d in decisions if d.merge]
        report["clusters_confirmed"] += len(confirmed)

        if confirmed:
            canon_atoms, merged_atoms, new_edges = _apply_merges(store, confirmed)

            # Re-embed canonical atoms — their context just changed.
            if canon_atoms:
                from ccchain.core.embedding import embed
                texts = [a.context for a in canon_atoms]
                embs = embed(
                    texts, base_url=config.embedder_base_url,
                    model=config.embedder_model, api_key=config.llm_api_key,
                )
                for a, e in zip(canon_atoms, embs):
                    a.embedding = e

            # Delete OLD edges touching dups BEFORE upsert (upsert is INSERT OR
            # IGNORE; without this, stale dup edges would linger / parallel-stack).
            dup_ids = [d for dec in confirmed for d in dec.dup_ids]
            store.delete_edges_touching(dup_ids)
            store.upsert_atoms(canon_atoms + merged_atoms)
            store.upsert_edges(new_edges)
            store.rebuild_graph()  # one rebuild per level

            report["atoms_merged"] += sum(len(d.dup_ids) for d in confirmed)
            report["canonical_ids"].extend(d.canonical_id for d in confirmed)

        report["per_level"][level] = {
            "candidates": len(all_clusters),
            "merged": sum(len(d.dup_ids) for d in confirmed),
        }
        report["levels_processed"].append(level)

    _progress("consolidating", 1.0)
    return report
