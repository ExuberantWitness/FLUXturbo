"""LeapRefiner — iterative gatekeeper validation + LLM fix loop.

Calls core.gatekeeper.validate() → feeds errors back to LLM → re-validates.
Max 3 rounds by default. v0.3: prompts instruct LLM to populate missing provenance
when R6 fires for CoE-triggering types.
"""

from __future__ import annotations

from ccchain.core.gatekeeper import apply_r6_demotions, validate
from ccchain.core.ontology import Atom, Edge
from ccchain.plugins.base import Refiner


class LeapRefiner(Refiner):
    """Iterative validation → LLM fix → re-validation loop."""

    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def refine(
        self,
        atoms: list[Atom],
        edges: list[Edge],
        segments: list[str] | None = None,
        max_rounds: int = 3,
    ) -> tuple[list[Atom], list[Edge], dict]:
        fix_log: dict = {"rounds": 0, "errors_found": [], "errors_fixed": []}

        current_atoms = list(atoms)
        current_edges = list(edges)

        for round_num in range(max_rounds):
            errors = validate(current_atoms, current_edges)
            if not errors:
                fix_log["rounds"] = round_num
                return current_atoms, current_edges, fix_log

            fix_log["errors_found"].append(len(errors))
            fix_log["rounds"] = round_num + 1

            # Only fix R1/R2/R3/R4/R6/R7 errors via LLM (R5 dedup is handled automatically)
            fixable = [e for e in errors if e["rule"] != "R5"]
            if not fixable:
                # Only dedup errors remain — merge duplicates
                current_atoms, current_edges = self._auto_dedup(current_atoms, current_edges, errors)
                fix_log["errors_fixed"].append(len(errors))
                continue

            current_atoms, current_edges = self._llm_fix(
                current_atoms, current_edges, fixable, segments
            )

            post_errors = validate(current_atoms, current_edges)
            new_count = len(post_errors)
            fixed_count = len(errors) - new_count
            fix_log["errors_fixed"].append(max(0, fixed_count))

            if not post_errors:
                fix_log["rounds"] = round_num + 1
                break

        # After max rounds: apply R6 demotions as final fallback (mutates remaining failing atoms)
        final_errors = validate(current_atoms, current_edges)
        if any(e["rule"] == "R6" for e in final_errors):
            apply_r6_demotions(current_atoms)

        return current_atoms, current_edges, fix_log

    def _llm_fix(
        self,
        atoms: list[Atom],
        edges: list[Edge],
        errors: list[dict],
        segments: list[str] | None,
    ) -> tuple[list[Atom], list[Edge]]:
        from ccchain.core.llm import chat_json

        atoms_json = [a.to_dict() for a in atoms]
        edges_json = [e.to_dict() for e in edges]
        errors_text = "\n".join(
            f"- [{e['rule']}] {e['target']}: {e['message']}" for e in errors
        )

        prompt = _FIX_PROMPT.format(
            atoms_json=atoms_json,
            edges_json=edges_json,
            errors_text=errors_text,
            original_text="\n".join(segments) if segments else "(not provided)",
        )

        response = chat_json(
            [{"role": "user", "content": prompt}],
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )

        return self._parse_fix_response(response, atoms, edges)

    def _parse_fix_response(
        self, response: dict, original_atoms: list[Atom], original_edges: list[Edge]
    ) -> tuple[list[Atom], list[Edge]]:
        fixed_atoms = list(original_atoms)
        fixed_edges = list(original_edges)

        for action in response.get("fixes", []):
            action_type = action.get("action", "")

            if action_type == "update_atom":
                node_id = action.get("node_id", "")
                for a in fixed_atoms:
                    if a.node_id == node_id:
                        if "type" in action:
                            a.type = action["type"]
                        if "level" in action:
                            a.level = action["level"]
                        if "context" in action:
                            a.context = action["context"]
                        if "provenance" in action and action["provenance"]:
                            # Merge new provenance keys over existing
                            if a.provenance is None:
                                a.provenance = {}
                            a.provenance.update(action["provenance"])

            elif action_type == "update_edge":
                src = action.get("src", "")
                tgt = action.get("tgt", "")
                for e in fixed_edges:
                    if e.src == src and e.tgt == tgt:
                        if "relation" in action:
                            e.relation = action["relation"]
                        if "rho" in action and action["rho"]:
                            from ccchain.core.ontology import Rho

                            e.rho = Rho.from_dict(action["rho"])

            elif action_type == "remove_edge":
                src = action.get("src", "")
                tgt = action.get("tgt", "")
                fixed_edges = [
                    e for e in fixed_edges
                    if not (e.src == src and e.tgt == tgt)
                ]

        return fixed_atoms, fixed_edges

    @staticmethod
    def _auto_dedup(
        atoms: list[Atom], edges: list[Edge], errors: list[dict]
    ) -> tuple[list[Atom], list[Edge]]:
        """Merge duplicate atoms (same name + type + level)."""
        dedup_errors = [e for e in errors if e["rule"] == "R5"]
        if not dedup_errors:
            return atoms, edges

        # Build map: name:type:level → list of node_ids
        groups: dict[tuple[str, str, str], list[str]] = {}
        for a in atoms:
            key = (a.name.lower(), a.type, a.level)
            groups.setdefault(key, []).append(a.node_id)

        keep_ids: set[str] = set()
        merge_map: dict[str, str] = {}  # duplicate_id → canonical_id

        for key, ids in groups.items():
            if len(ids) > 1:
                canonical = ids[0]
                keep_ids.add(canonical)
                for dup_id in ids[1:]:
                    merge_map[dup_id] = canonical
            else:
                keep_ids.add(ids[0])

        # Remove duplicate atoms
        fixed_atoms = [a for a in atoms if a.node_id not in merge_map]

        # Rewrite edge endpoints
        fixed_edges = list(edges)
        for e in fixed_edges:
            if e.src in merge_map:
                e.src = merge_map[e.src]
            if e.tgt in merge_map:
                e.tgt = merge_map[e.tgt]

        return fixed_atoms, fixed_edges


_FIX_PROMPT = """\
You are fixing validation errors in a knowledge graph of research atoms and edges.

CURRENT ATOMS:
{atoms_json}

CURRENT EDGES:
{edges_json}

VALIDATION ERRORS:
{errors_text}

ORIGINAL TEXT (for context):
{original_text}

Fix the atoms and edges to resolve ALL errors. Common fixes:
- R6: For numerical/citation/method/solution/experiment atoms missing provenance, populate it.
  numerical → provenance: {{"score": <float>, "score_std": <float|null>}}
  citation → provenance: {{"raw_citation": "<bibliographic string>"}}
  method/solution/experiment → provenance: {{"code_span": "<location>", "source_chunk": <int>}}
- R7: If type and level mismatch, change either so they are consistent (type→level via TYPE_TO_LEVEL).
- R2: If edge endpoints have incompatible types, change one atom's type.

Return JSON with a "fixes" array:
{{
  "fixes": [
    {{"action": "update_atom", "node_id": "...", "type": "...", "level": "...", "provenance": {{"score": 0.5}}}},
    {{"action": "update_edge", "src": "...", "tgt": "...", "relation": "...", "rho": {{"bottleneck": "...", "mechanism": "...", "tradeoff": "...", "confidence": 0.8}}}},
    {{"action": "remove_edge", "src": "...", "tgt": "..."}}
  ]
}}
"""
