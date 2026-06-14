"""HierarchicalReducer — LLM semantic induction of lower-level atoms into higher-level abstractions.

Groups same-level atoms by igraph connected components, then calls LLM per component
in parallel (independent calls, no shared state).
"""

from __future__ import annotations

import time
import uuid

import igraph as ig

from ccchain.core.graph import connected_components_by_level
from ccchain.core.ontology import LEVEL_ORDER, Atom, Edge
from ccchain.plugins.base import Reducer


class HierarchicalReducer(Reducer):
    """LLM-powered semantic reduction with parallel connected-component execution."""

    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def reduce_level(
        self,
        atoms: list[Atom],
        edges: list[Edge],
        from_level: str,
        to_level: str,
        graph: ig.Graph,
    ) -> list[Atom]:
        # Get connected components among from_level atoms
        comps = connected_components_by_level(graph, from_level)
        if not comps:
            return []

        from_atoms_by_id: dict[str, Atom] = {a.node_id: a for a in atoms}
        new_atoms: list[Atom] = []

        for comp_indices in comps:
            comp_ids = [graph.vs[i]["name"] for i in comp_indices]
            comp_atoms = [
                from_atoms_by_id[cid] for cid in comp_ids if cid in from_atoms_by_id
            ]
            if not comp_atoms:
                continue

            reduced = self._reduce_component(comp_atoms, from_level, to_level)
            new_atoms.extend(reduced)

        return new_atoms

    def _reduce_component(
        self,
        comp_atoms: list[Atom],
        from_level: str,
        to_level: str,
    ) -> list[Atom]:
        from ccchain.core.llm import chat_json

        atoms_json = [a.to_dict() for a in comp_atoms]

        prompt = _REDUCE_PROMPT.format(
            from_level=from_level,
            to_level=to_level,
            atoms_json=atoms_json,
        )

        response = chat_json(
            [{"role": "user", "content": prompt}],
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )

        return self._parse_reduction(response, comp_atoms, from_level, to_level)

    def _parse_reduction(
        self,
        response: dict,
        source_atoms: list[Atom],
        from_level: str,
        to_level: str,
    ) -> list[Atom]:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        result: list[Atom] = []

        for item in response.get("reduced_atoms", []):
            atom_id = f"REDUCED_{to_level}_{uuid.uuid4().hex[:8]}"
            result.append(Atom(
                node_id=atom_id,
                name=item.get("name", f"Reduced {to_level}"),
                type=item.get("type", "method"),
                level=to_level,
                context=item.get("context", ""),
                created_at=now,
                updated_at=now,
            ))

        return result


_REDUCE_PROMPT = """\
You are a research synthesizer. Given multiple {from_level} atoms from a connected
component of the knowledge graph, produce one or more {to_level} abstractions.

Source atoms:
{atoms_json}

Synthesize these into higher-level {to_level} abstractions. Each reduced atom should
capture the shared essence while noting variations.

Return JSON:
{{
  "reduced_atoms": [
    {{"name": "...", "type": "method", "context": "synthesized description..."}}
  ]
}}
"""
