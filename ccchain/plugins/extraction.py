"""TwoPhaseExtractor — LEAP-style two-phase extraction from text segments.

Phase 1: Extract W2 (problem analysis) + W3 (solution directions) — abstract layer.
Phase 2: Extract W4 (concrete solutions) + W5 (code implementations) — concrete layer,
         conditioned on Phase 1 results.
"""

from __future__ import annotations

import time
import uuid

from ccchain.core.ontology import (
    ATOM_TYPES,
    BOTTLENECK_CATEGORIES,
    CC_EDGE_TYPES,
    Atom,
    Edge,
    Rho,
)
from ccchain.plugins.base import Extractor


class TwoPhaseExtractor(Extractor):
    """LEAP Blueprint → Decompose extraction pipeline."""

    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def extract(
        self, segments: list[str], source_pdf: str
    ) -> tuple[list[Atom], list[Edge]]:
        combined = "\n\n--- chunk ---\n\n".join(segments)

        # Phase 1: W2 + W3
        w2w3_atoms, w2w3_edges = self._extract_phase1(combined, source_pdf)

        # Phase 2: W4 + W5 (conditioned on Phase 1)
        w4w5_atoms, w4w5_edges = self._extract_phase2(combined, source_pdf, w2w3_atoms)

        all_atoms = w2w3_atoms + w4w5_atoms
        all_edges = w2w3_edges + w4w5_edges
        return all_atoms, all_edges

    # ------------------------------------------------------------------
    # Phase 1: W2 + W3
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
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )

        return self._parse_phase1(response, source_pdf)

    def _parse_phase1(
        self, response: dict, source_pdf: str
    ) -> tuple[list[Atom], list[Edge]]:
        atoms: list[Atom] = []
        edges: list[Edge] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        w2_data = response.get("W2_problem_analysis", {})
        if w2_data:
            w2_id = self._id("W2", w2_data.get("name", "problem"), source_pdf)
            atoms.append(Atom(
                node_id=w2_id,
                name=w2_data.get("name", "Untitled Problem"),
                type="bottleneck",
                level="W2_problem_analysis",
                context=w2_data.get("context", ""),
                source_pdf=source_pdf,
                created_at=now,
                updated_at=now,
            ))

            for w3_raw in response.get("W3_solution_directions", []):
                w3_id = self._id("W3", w3_raw.get("name", "solution"), source_pdf)
                atoms.append(Atom(
                    node_id=w3_id,
                    name=w3_raw.get("name", "Untitled Direction"),
                    type="method",
                    level="W3_solution_direction",
                    context=w3_raw.get("context", ""),
                    source_pdf=source_pdf,
                    created_at=now,
                    updated_at=now,
                ))
                edges.append(Edge(
                    src=w2_id,
                    relation="decomposes_into",
                    tgt=w3_id,
                ))

                # Cross-direction comparisons
                for comp in w3_raw.get("compares_to", []):
                    comp_id = self._id("W3", comp, source_pdf)
                    edges.append(Edge(
                        src=w3_id,
                        relation="compares",
                        tgt=comp_id,
                    ))

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
            for a in phase1_atoms
            if a.level == "W3_solution_direction"
        )

        prompt = _PHASE2_PROMPT.format(
            text=text[:12000],
            w3_summary=w3_summary or "(none)",
        )

        response = chat_json(
            [{"role": "user", "content": prompt}],
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )

        return self._parse_phase2(response, source_pdf)

    def _parse_phase2(
        self, response: dict, source_pdf: str
    ) -> tuple[list[Atom], list[Edge]]:
        atoms: list[Atom] = []
        edges: list[Edge] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        for w4_raw in response.get("W4_concrete_solutions", []):
            w4_id = self._id("W4", w4_raw.get("name", "solution"), source_pdf)
            atoms.append(Atom(
                node_id=w4_id,
                name=w4_raw.get("name", "Untitled Solution"),
                type="method",
                level="W4_concrete_solution",
                context=w4_raw.get("context", ""),
                source_pdf=source_pdf,
                created_at=now,
                updated_at=now,
            ))

            # Link to parent W3
            parent_w3 = w4_raw.get("parent_W3_id", "")
            edges.append(Edge(
                src=w4_id,
                relation="aggregates_to",
                tgt=parent_w3 if parent_w3 else "",
            ))

            # W5 children
            for w5_raw in w4_raw.get("W5_implementations", []):
                w5_id = self._id("W5", w5_raw.get("name", "impl"), source_pdf)
                code_ref = w5_raw.get("code_ref", "")
                atoms.append(Atom(
                    node_id=w5_id,
                    name=w5_raw.get("name", "Untitled Implementation"),
                    type="component",
                    level="W5_code_implementation",
                    context=w5_raw.get("context", ""),
                    code_ref=code_ref if code_ref else None,
                    source_pdf=source_pdf,
                    created_at=now,
                    updated_at=now,
                ))
                edges.append(Edge(
                    src=w4_id,
                    relation="decomposes_into",
                    tgt=w5_id,
                ))

            # Cross-solution edges
            for ext in w4_raw.get("extends", []):
                edges.append(Edge(
                    src=w4_id,
                    relation="extends",
                    tgt=ext,
                    rho=_parse_rho(w4_raw.get("extends_rho", {})),
                ))
            for imp in w4_raw.get("improves", []):
                edges.append(Edge(
                    src=w4_id,
                    relation="improves",
                    tgt=imp,
                    rho=_parse_rho(w4_raw.get("improves_rho", {})),
                ))

        return atoms, edges

    @staticmethod
    def _id(prefix: str, name: str, source_pdf: str) -> str:
        slug = name.lower().replace(" ", "_")[:40]
        short_uuid = uuid.uuid4().hex[:8]
        pdf_tag = source_pdf.replace(".pdf", "").replace(" ", "_")[:20] if source_pdf else "doc"
        return f"{prefix}_{slug}_{pdf_tag}_{short_uuid}"


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
# Prompts
# ---------------------------------------------------------------------------
_PHASE1_PROMPT = """\
You are a research analyst extracting structured knowledge from an academic paper.

Extract:
1. W2_problem_analysis: The core problem/bottleneck this paper addresses. Exactly ONE.
   Categorize using: {bottleneck_categories}
2. W3_solution_directions: The high-level solution approaches proposed (2-5 items).
   Include "compares_to" for alternative directions mentioned/discussed in the paper.

For each atom, provide: name (short label), context (1-2 sentence summary from the text).

Return JSON:
{{
  "W2_problem_analysis": {{"name": "...", "context": "..."}},
  "W3_solution_directions": [
    {{"name": "...", "context": "...", "compares_to": ["alt_name_1"]}}
  ]
}}

Paper text:
{text}
"""

_PHASE2_PROMPT = """\
You are a research analyst decomposing solution directions into concrete implementations.

Phase 1 identified these W3 solution directions:
{w3_summary}

Now extract:
1. W4_concrete_solutions: Specific methods/algorithms. For each, include:
   - name, context (what it does and how)
   - parent_W3_id: the W3 name this solution belongs to
   - extends: names of prior methods this extends
   - improves: names of prior methods this improves upon
   - extends_rho/improves_rho: {{"bottleneck": "...", "mechanism": "...", "tradeoff": "...", "confidence": 0.8}}

2. W5_implementations: Code-level details for each W4. Include name, context, code_ref (function/class name).

Return JSON:
{{
  "W4_concrete_solutions": [
    {{
      "name": "...",
      "context": "...",
      "parent_W3_id": "...",
      "extends": ["prior_method_name"],
      "improves": ["prior_method_name"],
      "extends_rho": {{"bottleneck": "...", "mechanism": "...", "tradeoff": "...", "confidence": 0.8}},
      "improves_rho": {{"bottleneck": "...", "mechanism": "...", "tradeoff": "...", "confidence": 0.8}},
      "W5_implementations": [
        {{"name": "...", "context": "...", "code_ref": "function_name"}}
      ]
    }}
  ]
}}

Paper text:
{text}
"""
