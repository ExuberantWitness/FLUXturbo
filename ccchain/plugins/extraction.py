"""TwoPhaseExtractor — LEAP-style two-phase extraction from text segments.

Phase 1: Extract the top tiers W1 (problem) + W2 (direction) + W3 (approach).
Phase 2: Extract the bottom tiers W4 (implementation) + W5 (code),
         conditioned on Phase 1's W3 approaches.

v0.5: type and level are DECOUPLED — each atom gets an independently-chosen
level (W1..W5) and type (12-type vocabulary). Prompts list both axes and let
the LLM pick the best fit. Provenance is still populated for CoE-triggering
types (numerical/citation/method/solution/experiment).
"""

from __future__ import annotations

import time
import uuid

from ccchain.core.ontology import (
    ATOM_TYPE_SET,
    BOTTLENECK_CATEGORIES,
    LEVEL_DEFAULT_TYPE,
    Atom,
    Edge,
    Rho,
)
from ccchain.plugins.base import Extractor


class TwoPhaseExtractor(Extractor):
    """LEAP Blueprint → Decompose extraction pipeline (5 levels, decoupled type)."""

    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def extract(
        self, segments: list[str], source_pdf: str
    ) -> tuple[list[Atom], list[Edge]]:
        combined = "\n\n--- chunk ---\n\n".join(segments)

        # Phase 1: W1 + W2 + W3 (top tiers)
        p1_atoms, p1_edges = self._extract_phase1(combined, source_pdf)

        # Phase 2: W4 + W5 (bottom tiers, conditioned on Phase 1)
        p2_atoms, p2_edges = self._extract_phase2(combined, source_pdf, p1_atoms)

        return p1_atoms + p2_atoms, p1_edges + p2_edges

    # ------------------------------------------------------------------
    # Phase 1: W1 + W2 + W3
    # ------------------------------------------------------------------
    def _extract_phase1(
        self, text: str, source_pdf: str
    ) -> tuple[list[Atom], list[Edge]]:
        from ccchain.core.llm import chat_json

        prompt = _PHASE1_PROMPT.format(
            text=text[:12000],
            bottleneck_categories=", ".join(BOTTLENECK_CATEGORIES),
        )
        response = chat_json(
            [{"role": "user", "content": prompt}],
            base_url=self.base_url, api_key=self.api_key, model=self.model,
        )
        return self._parse_phase1(response, source_pdf)

    def _parse_phase1(
        self, response: dict, source_pdf: str
    ) -> tuple[list[Atom], list[Edge]]:
        atoms: list[Atom] = []
        edges: list[Edge] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        def _make(raw: dict, level: str, fallback_type: str, prefix: str) -> Atom:
            t = self._valid_type(raw.get("type"), fallback_type)
            prov = dict(raw.get("provenance") or {"phase": "extract", "via": "phase1"})
            if t == "citation" and "raw_citation" not in prov:
                prov["raw_citation"] = (raw.get("context") or "")[:300]
            if t == "numerical" and "score" not in prov:
                prov["score"] = raw.get("score")
            return Atom(
                node_id=self._id(prefix, raw.get("name", level), source_pdf),
                name=raw.get("name", "Untitled"),
                type=t, level=level,
                context=raw.get("context", ""),
                source_pdf=source_pdf, created_at=now, updated_at=now,
                provenance=prov,
            )

        # W1 problem (exactly one) → W2 directions → W3 approaches
        w1_raw = response.get("W1_problem", {})
        if w1_raw:
            w1 = _make(w1_raw, "W1_problem", "problem", "W1")
            atoms.append(w1)

            for w2_raw in response.get("W2_directions", []):
                w2 = _make(w2_raw, "W2_direction", "method", "W2")
                atoms.append(w2)
                edges.append(Edge(src=w1.node_id, relation="decomposes_into", tgt=w2.node_id))
                edges.append(Edge(src=w2.node_id, relation="aggregates_to", tgt=w1.node_id))

                for w3_raw in w2_raw.get("W3_approaches", []):
                    w3 = _make(w3_raw, "W3_approach", "method", "W3")
                    atoms.append(w3)
                    edges.append(Edge(src=w2.node_id, relation="decomposes_into", tgt=w3.node_id))
                    edges.append(Edge(src=w3.node_id, relation="aggregates_to", tgt=w2.node_id))
                    # cross-direction comparisons
                    for comp in w3_raw.get("compares_to", []):
                        edges.append(Edge(src=w3.node_id, relation="compares",
                                          tgt=self._id("W3", comp, source_pdf)))

        return atoms, edges

    # ------------------------------------------------------------------
    # Phase 2: W4 + W5
    # ------------------------------------------------------------------
    def _extract_phase2(
        self, text: str, source_pdf: str, phase1_atoms: list[Atom]
    ) -> tuple[list[Atom], list[Edge]]:
        from ccchain.core.llm import chat_json

        w3_summary = "\n".join(
            f"- {a.name}: {a.context[:200]}"
            for a in phase1_atoms if a.level == "W3_approach"
        )
        prompt = _PHASE2_PROMPT.format(text=text[:12000], w3_summary=w3_summary or "(none)")
        response = chat_json(
            [{"role": "user", "content": prompt}],
            base_url=self.base_url, api_key=self.api_key, model=self.model,
        )
        return self._parse_phase2(response, source_pdf, phase1_atoms)

    def _parse_phase2(
        self, response: dict, source_pdf: str, phase1_atoms: list[Atom] | None = None
    ) -> tuple[list[Atom], list[Edge]]:
        atoms: list[Atom] = []
        edges: list[Edge] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        for w4_raw in response.get("W4_implementations", []):
            w4_id = self._id("W4", w4_raw.get("name", "impl"), source_pdf)
            w4_type = self._valid_type(w4_raw.get("type"), "solution")
            w4_prov = dict(w4_raw.get("provenance") or {"phase": "extract", "via": "phase2"})
            if w4_type == "numerical" and "score" not in w4_prov:
                w4_prov["score"] = w4_raw.get("score")
            atoms.append(Atom(
                node_id=w4_id,
                name=w4_raw.get("name", "Untitled Implementation"),
                type=w4_type, level="W4_implementation",
                context=w4_raw.get("context", ""),
                source_pdf=source_pdf, created_at=now, updated_at=now,
                provenance=w4_prov,
            ))

            # Link W4 → its parent W3 (resolve name → node_id, avoid dangling edge)
            parent_w3_id = self._resolve_parent_id(
                w4_raw.get("parent_W3_id", ""), phase1_atoms, "W3_approach"
            )
            if parent_w3_id:
                edges.append(Edge(src=parent_w3_id, relation="decomposes_into", tgt=w4_id))
                edges.append(Edge(src=w4_id, relation="aggregates_to", tgt=parent_w3_id))

            # W5 children
            for w5_raw in w4_raw.get("W5_code", []):
                w5_id = self._id("W5", w5_raw.get("name", "code"), source_pdf)
                code_ref = w5_raw.get("code_ref", "")
                w5_type = self._valid_type(w5_raw.get("type"), "component")
                w5_prov = w5_raw.get("provenance")
                if w5_type == "experiment" and not w5_prov:
                    w5_prov = {"phase": "extract", "via": "phase2",
                               "code_span": code_ref or "unknown"}
                atoms.append(Atom(
                    node_id=w5_id,
                    name=w5_raw.get("name", "Untitled Code"),
                    type=w5_type, level="W5_code",
                    context=w5_raw.get("context", ""),
                    code_ref=code_ref if code_ref else None,
                    code_body=w5_raw.get("code_body"),
                    source_pdf=source_pdf, created_at=now, updated_at=now,
                    provenance=w5_prov,
                ))
                edges.append(Edge(src=w4_id, relation="decomposes_into", tgt=w5_id))

            # cross-solution edges
            for ext in w4_raw.get("extends", []):
                edges.append(Edge(src=w4_id, relation="extends", tgt=ext,
                                  rho=_parse_rho(w4_raw.get("extends_rho", {}))))
            for imp in w4_raw.get("improves", []):
                edges.append(Edge(src=w4_id, relation="improves", tgt=imp,
                                  rho=_parse_rho(w4_raw.get("improves_rho", {}))))

        return atoms, edges

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _valid_type(raw: str | None, fallback: str) -> str:
        """Validate the LLM-chosen type against the 12-type vocab; fall back if invalid.

        v0.5: type is decoupled from level, so we only check type validity — the
        level is set by the caller based on which tier the atom belongs to.
        """
        if raw and raw in ATOM_TYPE_SET:
            return raw
        return fallback

    @staticmethod
    def _id(prefix: str, name: str, source_pdf: str) -> str:
        slug = name.lower().replace(" ", "_")[:40]
        short_uuid = uuid.uuid4().hex[:8]
        pdf_tag = source_pdf.replace(".pdf", "").replace(" ", "_")[:20] if source_pdf else "doc"
        return f"{prefix}_{slug}_{pdf_tag}_{short_uuid}"

    @staticmethod
    def _resolve_parent_id(
        parent_ref: str, phase1_atoms: list[Atom] | None, target_level: str
    ) -> str:
        """Resolve a parent reference (usually the parent *name*) to its node_id.

        Phase 2 asks the LLM for the parent W3 by name, but edges need the node_id.
        Without this the W3→W4 link dangles and gatekeeper R1 drops it — breaking
        the W1→W2→W3→W4→W5 chain.
        """
        if not parent_ref or not phase1_atoms:
            return ""
        ref = parent_ref.strip().lower()
        candidates = [a for a in phase1_atoms if a.level == target_level]
        for a in candidates:
            if a.name.strip().lower() == ref:
                return a.node_id
        for a in candidates:
            n = a.name.strip().lower()
            if n.startswith(ref) or ref.startswith(n):
                return a.node_id
        return ""


def _parse_rho(data: dict) -> Rho | None:
    if not data:
        return None
    return Rho(
        bottleneck=str(data.get("bottleneck", "")),
        mechanism=str(data.get("mechanism", "")),
        tradeoff=str(data.get("tradeoff", "")),
        confidence=float(data.get("confidence", 0.7)),
    )


# ---------------------------------------------------------------------------
# Prompts (v0.5: level and type are independent axes)
# ---------------------------------------------------------------------------
_PHASE1_PROMPT = """\
You are a research analyst extracting structured knowledge from an academic paper.

The knowledge pyramid has 5 abstraction tiers; for each atom pick BOTH a level
AND a type independently:

  LEVELS (abstraction tier):
    W1_problem        — the problem/bottleneck the paper addresses
    W2_direction      — the research direction(s) pursued
    W3_approach       — the core idea/思路 (the specific technical approach)
  TYPES (12-type vocabulary, usable at ANY level):
    problem, bottleneck, hypothesis, method, citation, concept,
    solution, numerical, conclusion, component, experiment, verification

Extract (top tiers):
1. W1_problem: The core problem/bottleneck. EXACTLY ONE. If type "bottleneck",
   categorize using: {bottleneck_categories}
2. W2_directions: The research direction(s) (1-3 items).
3. For each W2 direction, W3_approaches: the core idea(s)/思路 under it (1-3 each).

For each atom provide: name (short label), context (1-2 sentence summary),
type (best fit from the 12 types), and provenance when relevant:
  citation → {{"raw_citation": "<full bibliographic string>"}}
  numerical → {{"score": <float>, "score_std": <float|null>}}
  method/solution/experiment → {{"code_span": "...", "source_chunk": <int>}}
Include "compares_to" (names) for alternatives the paper discusses.

Return JSON:
{{
  "W1_problem": {{"name": "...", "context": "...", "type": "bottleneck"}},
  "W2_directions": [
    {{"name": "...", "context": "...", "type": "method", "provenance": {{"code_span": "..."}},
      "compares_to": ["alt_name"],
      "W3_approaches": [
        {{"name": "...", "context": "...", "type": "method", "provenance": {{...}}}}
      ]}}
  ]
}}

Paper text:
{text}
"""

_PHASE2_PROMPT = """\
You are a research analyst decomposing approaches into concrete implementations.

Phase 1 identified these W3 approaches (思路):
{w3_summary}

Now extract the bottom tiers:
  W4_implementation  — concrete designs / algorithms / numerical results
  W5_code            — code-level details

LEVELS & TYPES (independent — pick best fit for each atom):
  LEVELS: W4_implementation, W5_code
  TYPES (12-type vocabulary, usable at ANY level):
    problem, bottleneck, hypothesis, method, citation, concept,
    solution, numerical, conclusion, component, experiment, verification

For each W4 implementation:
  - name, context (what it does and how), type
  - parent_W3_id: the W3 approach NAME this implements
  - extends / improves: names of prior methods (with extends_rho / improves_rho)
  - provenance when relevant:
      numerical → {{"score": <float>, "score_std": <float|null>}}

For each W5 code item (nested under its W4):
  - name, context, type (typically component/experiment/verification)
  - code_ref (function/class name), code_body (full text if available)
  - experiment → provenance {{"code_span": "...", "source_chunk": <int>}}

Return JSON:
{{
  "W4_implementations": [
    {{
      "name": "...", "context": "...", "type": "solution",
      "parent_W3_id": "...",
      "provenance": {{...}},
      "extends": ["prior_name"], "improves": ["prior_name"],
      "extends_rho": {{"bottleneck": "...", "mechanism": "...", "tradeoff": "...", "confidence": 0.8}},
      "improves_rho": {{"bottleneck": "...", "mechanism": "...", "tradeoff": "...", "confidence": 0.8}},
      "W5_code": [
        {{"name": "...", "context": "...", "type": "component", "code_ref": "fn", "code_body": "..."}}
      ]
    }}
  ]
}}

Paper text:
{text}
"""
