"""BlueprintExtractor: LLM-driven CC atom + edge extraction with OntologyGatekeeper validation."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from .ontology import (
    ATOM_TYPES,
    BOTTLENECK_CATEGORIES,
    EDGE_TYPES,
    STRONG_CAUSAL,
    Atom,
    Blueprint,
    Edge,
    OntologyGatekeeper,
    Rho,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (LEAP: plan-first, Flux-Insight: CC type constraints)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an academic knowledge-graph extractor. Your task has two phases:
  Phase 1 — Identify key concepts as CC Atoms.
  Phase 2 — Extract relationships between those atoms as CC Edges.

## CC Atom Types (you MUST only use these):
  method       — algorithm / model / architecture  (e.g. "ReMAC", "Transformer")
  bottleneck   — problem / challenge / limitation   (e.g. "credit assignment problem")
  paper        — publication reference              (e.g. "Yu et al. 2022")
  fact         — experimental fact / numeric result  (e.g. "93.2% on HalfCheetah")
  component    — module / sub-component             (e.g. "centralized critic")
  hypothesis   — assumption / claim                 (e.g. "shared params improve efficiency")
  experiment   — experimental setup / evaluation    (e.g. "ablation study on 3 agents")
  verification — proof / evidence / validation      (e.g. "convergence proof in Appendix A")

## CC Edge Types (you MUST only use these):
  Strong causal (require a "rho" evidence record):
    EXTENDS        — A extends B (builds upon)
    IMPROVES       — A improves upon B
    REPLACES       — A replaces B
    ADAPTS         — A adapts B to a new setting
  Weak:
    USES_COMPONENT — A uses B as a component
    COMPARES       — A is compared against B
  Semantic:
    BACKGROUND     — A is background knowledge for B
    IMPLEMENTS     — A implements B (theory → implementation)
    VALIDATES      — A validates B (experiment validates hypothesis)
    BOUNDARY_OF    — A defines a boundary / scope of B
    RELATED_TO     — general association (fallback)

## Rho evidence record (REQUIRED for EXTENDS / IMPROVES / REPLACES / ADAPTS):
  bottleneck: which bottleneck does this edge address? (use one of the bottleneck categories or a specific one from the text)
  mechanism:  how does the source address this bottleneck?
  tradeoff:   what cost or limitation does this introduce?
  confidence: 0.0–1.0 how confident is this relationship?

## Bottleneck categories (prefer these for rho.bottleneck):
  {bottleneck_list}

## Output format — return ONLY valid JSON:
{{
  "atoms": [
    {{"name": "...", "type": "method", "context": "exact snippet from text"}}
  ],
  "edges": [
    {{
      "src": "atom_name", "relation": "IMPROVES", "tgt": "atom_name",
      "rho": {{"bottleneck": "...", "mechanism": "...", "tradeoff": "...", "confidence": 0.8}}
    }}
  ]
}}

## Rules:
1. Atom name MUST be >= 2 characters. Reject single letters and pure numbers.
2. Prefer concrete names ("ReMAC" not "the algorithm", "MAPPO" not "baseline method").
3. Each atom's name MUST appear (or be clearly derivable) from the source text.
4. Only extract edges where BOTH endpoints exist in the atoms list.
5. Strong-causal edges (EXTENDS/IMPROVES/REPLACES/ADAPTS) MUST include rho.
6. Non-causal edges should NOT include rho (omit the field).
7. Keep atom names consistent — use the same string for references in edges.
8. Extract at most {max_atoms} atoms and {max_edges} edges from this text.
""".strip()


def _build_system_prompt(max_atoms: int = 30, max_edges: int = 20) -> str:
    return SYSTEM_PROMPT.format(
        bottleneck_list=", ".join(BOTTLENECK_CATEGORIES),
        max_atoms=max_atoms,
        max_edges=max_edges,
    )


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response (handles ```json fences and raw JSON)."""
    m = _JSON_BLOCK_RE.search(text)
    payload = m.group(1).strip() if m else text.strip()
    return json.loads(payload)


# ---------------------------------------------------------------------------
# BlueprintExtractor
# ---------------------------------------------------------------------------
class BlueprintExtractor:
    def __init__(
        self,
        llm_model_name: str = "deepseek-chat",
        llm_base_url: str = "https://api.deepseek.com",
        api_key: str = "",
        http_proxy: str = "",
        max_atoms: int = 30,
        max_edges: int = 20,
    ):
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.gatekeeper = OntologyGatekeeper()
        self.system_prompt = _build_system_prompt(max_atoms, max_edges)

        # Build OpenAI-compatible client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY", ""),
            "base_url": llm_base_url,
        }
        if http_proxy:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxy=http_proxy)
        self.client = OpenAI(**client_kwargs)
        self.model = llm_model_name

    # ---- single-segment extraction ----
    def extract_from_text(self, text: str, doc_idx: int = 0) -> Blueprint:
        """Extract CC atoms and edges from a single text segment via LLM."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return Blueprint()

        try:
            data = _parse_json_response(raw)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed: %s\nRaw: %.300s", exc, raw)
            return Blueprint()

        # Parse atoms
        atoms: list[Atom] = []
        for a_data in data.get("atoms", []):
            try:
                atom = Atom.from_dict(a_data)
                errs = self.gatekeeper.validate_atom(atom)
                if errs:
                    logger.warning("Skipping atom %s: %s", a_data.get("name"), errs)
                    continue
                if not self.gatekeeper.validate_entity(atom.name):
                    continue
                atoms.append(atom)
            except (ValueError, KeyError) as exc:
                logger.warning("Bad atom %s: %s", a_data, exc)

        atom_map = {a.name: a for a in atoms}

        # Parse edges
        edges: list[Edge] = []
        for e_data in data.get("edges", []):
            try:
                edge = Edge.from_dict(e_data)
                errs = self.gatekeeper.validate_edge(edge, atom_map)
                if errs:
                    # Drop Rho-less strong-causal edges rather than rejecting entirely
                    if any("Rho" in e for e in errs) and edge.relation in STRONG_CAUSAL:
                        continue
                    logger.warning("Edge issues %s→%s: %s", edge.src, edge.tgt, errs)
                if edge.src in atom_map and edge.tgt in atom_map:
                    edges.append(edge)
            except (ValueError, KeyError) as exc:
                logger.warning("Bad edge %s: %s", e_data, exc)

        bp = Blueprint(atoms=atoms, edges=edges)
        return bp

    # ---- batch extraction ----
    def extract_batch(
        self,
        segments: list[str],
        max_workers: int = 4,
    ) -> list[Blueprint]:
        """Extract blueprints from multiple segments in parallel."""
        results: list[Blueprint | None] = [None] * len(segments)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.extract_from_text, seg, idx): idx
                for idx, seg in enumerate(segments)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    logger.error("Segment %d failed: %s", idx, exc)
                    results[idx] = Blueprint()

        return [r or Blueprint() for r in results]
