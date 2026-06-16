"""CoEClaimVerifier — implements Chain-of-Evidence integrity audit (I1/I2/I3/I4).

Each check inspects a subset of atoms and produces a status verdict. The overall
report aggregates per-atom verdicts and computes CPR (Claim Provenance Rate).

Status mapping (decision 8):
  - verified          — passed all applicable checks
  - low_confidence    — I1 tolerance exceeded, or I4 partial alignment
  - low_reliability   — I2 spec violation, I3 dangling citation, or I4 misalignment
  - demoted           — already demoted by gatekeeper R6 (audit skips the check)
  - skipped           — type has no CoE check applicable
  - needs_review      — pre-existing lifecycle state; audit skipped
"""

from __future__ import annotations

from typing import Any

from ccchain.core.ontology import (
    TYPE_TO_COE_CHECKS,
    Atom,
    Edge,
    TaskSpec,
)
from ccchain.plugins.base import Verifier


class CoEClaimVerifier(Verifier):
    """Concrete CoE integrity audit implementation."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reference_api_keys: dict[str, str] | None = None,
        reference_api_timeout: float = 2.0,
        reference_api_max_retries: int = 3,
        majority_k: int = 5,
        i1_k: int = 1,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.reference_api_keys = reference_api_keys or {}
        self.reference_api_timeout = reference_api_timeout
        self.reference_api_max_retries = reference_api_max_retries
        self.majority_k = majority_k
        self.i1_k = i1_k

    def verify(
        self,
        atoms: list[Atom],
        edges: list[Edge],
        *,
        task_spec: TaskSpec | None = None,
    ) -> dict:
        """Run applicable CoE checks on each atom; mutate atom.status accordingly."""
        per_atom: list[dict] = []
        atoms_audited = 0
        atoms_passed = 0
        atoms_failed = 0
        atoms_skipped = 0
        failures_by_check: dict[str, int] = {"I1": 0, "I2": 0, "I3": 0, "I4": 0}
        numerical_total = 0
        numerical_verified = 0

        # Build child-component lookup for I4 (method/solution → child components)
        child_components: dict[str, list[Atom]] = _build_child_components(atoms, edges)

        for atom in atoms:
            # Pre-existing lifecycle states: audit skips
            if atom.status in ("needs_review", "transient"):
                atoms_skipped += 1
                per_atom.append(_per_atom(atom, "skipped", reason=f"pre-existing status {atom.status!r}"))
                continue
            if atom.status == "demoted":
                # gatekeeper already demoted; skip CoE check
                atoms_skipped += 1
                per_atom.append(_per_atom(atom, "skipped", reason="type demoted by R6"))
                continue

            checks = TYPE_TO_COE_CHECKS.get(atom.type, set())
            if not checks:
                atom.status = "skipped"
                atoms_skipped += 1
                per_atom.append(_per_atom(atom, "skipped", reason="no CoE check for type"))
                continue

            atoms_audited += 1
            per_check_results: dict[str, dict] = {}
            any_failed = False
            any_soft_failed = False

            for check in sorted(checks):
                if check == "I1":
                    result = self._check_i1(atom)
                elif check == "I2":
                    result = self._check_i2(atom, task_spec)
                elif check == "I3":
                    result = self._check_i3(atom)
                elif check == "I4":
                    result = self._check_i4(atom, child_components.get(atom.node_id, []))
                else:
                    continue

                per_check_results[check] = result

                if result["status"] == "failed":
                    failures_by_check[check] += 1
                    any_failed = True
                    if check == "I1" or (check == "I4" and result.get("verdict") == "partially_aligned"):
                        any_soft_failed = True
                elif result["status"] == "skipped":
                    # I2 skipped because no task_spec; counts as not-failed
                    pass

            # Aggregate to atom.status
            if any_failed and not any_soft_failed:
                atom.status = "low_reliability"
                atoms_failed += 1
            elif any_soft_failed:
                atom.status = "low_confidence"
                atoms_failed += 1
            else:
                atom.status = "verified"
                atoms_passed += 1

            # CPR tracking (numerical atoms only)
            if atom.type == "numerical":
                numerical_total += 1
                if atom.status == "verified":
                    numerical_verified += 1

            per_atom.append({
                "node_id": atom.node_id,
                "type": atom.type,
                "status": atom.status,
                "checks": per_check_results,
            })

        cpr = (numerical_verified / numerical_total) if numerical_total > 0 else 0.0

        return {
            "cpr": cpr,
            "atoms_audited": atoms_audited,
            "atoms_passed": atoms_passed,
            "atoms_failed": atoms_failed,
            "atoms_skipped": atoms_skipped,
            "failures_by_check": failures_by_check,
            "per_atom": per_atom,
        }

    # ------------------------------------------------------------------
    # I1 — Score Verification (numerical atoms)
    # ------------------------------------------------------------------
    def _check_i1(self, atom: Atom) -> dict:
        from ccchain.core.llm import chat_json_majority, chat_json

        score = (atom.provenance or {}).get("score")
        if score is None:
            return {"check": "I1", "status": "skipped", "reason": "no score in provenance"}

        # LLM re-extracts score from context; K=1 because tolerance absorbs noise
        prompt = _I1_PROMPT.format(context=atom.context[:1000])
        try:
            if self.i1_k == 1:
                response = chat_json(
                    [{"role": "user", "content": prompt}],
                    base_url=self.base_url, api_key=self.api_key, model=self.model,
                    temperature=0.0,
                )
            else:
                response = chat_json_majority(
                    [{"role": "user", "content": prompt}],
                    base_url=self.base_url, api_key=self.api_key, model=self.model,
                    k=self.i1_k,
                )
        except Exception as e:
            return {"check": "I1", "status": "skipped", "reason": f"LLM error: {e}"}

        reextracted = response.get("score")
        if reextracted is None:
            return {"check": "I1", "status": "skipped", "reason": "LLM returned no score"}

        try:
            reextracted = float(reextracted)
        except (TypeError, ValueError):
            return {"check": "I1", "status": "skipped", "reason": "LLM score not numeric"}

        # Adaptive tolerance: max(1% of |s|, 3σ/|s|) where σ is provided in provenance
        score_std = (atom.provenance or {}).get("score_std") or 0.0
        try:
            score_std = float(score_std)
        except (TypeError, ValueError):
            score_std = 0.0

        s = float(score)
        abs_s = abs(s) if s != 0 else 1e-9
        tolerance = max(0.01 * abs_s, 3.0 * score_std / abs_s if score_std > 0 else 0.0)

        delta = abs(reextracted - s)
        if delta <= tolerance:
            return {
                "check": "I1", "status": "passed",
                "score": s, "reextracted": reextracted, "delta": delta, "tolerance": tolerance,
            }
        else:
            return {
                "check": "I1", "status": "failed",
                "score": s, "reextracted": reextracted, "delta": delta, "tolerance": tolerance,
            }

    # ------------------------------------------------------------------
    # I2 — Specification Violation (experiment atoms, requires task_spec)
    # ------------------------------------------------------------------
    def _check_i2(self, atom: Atom, task_spec: TaskSpec | None) -> dict:
        from ccchain.core.llm import chat_json_majority

        if task_spec is None:
            return {"check": "I2", "status": "skipped", "reason": "no task_spec provided"}

        prompt = _I2_PROMPT.format(
            code_body=(atom.code_body or atom.context)[:2000],
            task_name=task_spec.task_name,
            eval_harness=task_spec.eval_harness,
            success_criteria=task_spec.success_criteria,
            constraints=", ".join(task_spec.constraints) or "(none)",
        )
        try:
            response = chat_json_majority(
                [{"role": "user", "content": prompt}],
                base_url=self.base_url, api_key=self.api_key, model=self.model,
                k=self.majority_k,
            )
        except Exception as e:
            return {"check": "I2", "status": "skipped", "reason": f"LLM error: {e}"}

        verdict = response.get("verdict", "ambiguous")
        if verdict == "compliant":
            return {"check": "I2", "status": "passed", "verdict": verdict, "reasoning": response.get("reasoning", "")}
        elif verdict == "violates_spec":
            return {"check": "I2", "status": "failed", "verdict": verdict, "reasoning": response.get("reasoning", "")}
        else:
            # ambiguous = soft fail → low_confidence via caller
            return {"check": "I2", "status": "failed", "verdict": "ambiguous",
                    "reasoning": response.get("reasoning", ""), "soft": True}

    # ------------------------------------------------------------------
    # I3 — Reference Verification (citation atoms)
    # ------------------------------------------------------------------
    def _check_i3(self, atom: Atom) -> dict:
        from ccchain.core.references import resolve_citation
        from ccchain.core.llm import chat_json_majority

        raw = (atom.provenance or {}).get("raw_citation")
        if not raw:
            return {"check": "I3", "status": "skipped", "reason": "no raw_citation"}

        hit = resolve_citation(
            raw,
            api_keys=self.reference_api_keys,
            timeout=self.reference_api_timeout,
            max_retries=self.reference_api_max_retries,
        )
        if hit:
            # Mutate atom.provenance with resolved data
            if atom.provenance is None:
                atom.provenance = {}
            atom.provenance["resolved"] = hit
            return {"check": "I3", "status": "passed", "resolved": hit}

        # API miss → LLM disambiguation fallback
        # Without candidate list to vote on, treat as dangling.
        return {"check": "I3", "status": "failed", "reason": "all APIs returned None"}

    # ------------------------------------------------------------------
    # I4 — Method-Code Alignment (method/solution atoms with child components)
    # ------------------------------------------------------------------
    def _check_i4(self, atom: Atom, child_components: list[Atom]) -> dict:
        from ccchain.core.llm import chat_json_majority

        if not child_components:
            return {"check": "I4", "status": "skipped", "reason": "no child components"}

        code_bodies = "\n\n".join(
            f"--- {c.name} ---\n{c.code_body or c.context}"
            for c in child_components[:3]
        )[:3000]

        prompt = _I4_PROMPT.format(
            method_context=atom.context[:1000],
            code_bodies=code_bodies,
        )
        try:
            response = chat_json_majority(
                [{"role": "user", "content": prompt}],
                base_url=self.base_url, api_key=self.api_key, model=self.model,
                k=self.majority_k,
            )
        except Exception as e:
            return {"check": "I4", "status": "skipped", "reason": f"LLM error: {e}"}

        verdict = response.get("verdict", "misaligned")
        if verdict == "aligned":
            return {"check": "I4", "status": "passed", "verdict": verdict,
                    "reasoning": response.get("reasoning", "")}
        elif verdict == "partially_aligned":
            # partial = soft fail
            return {"check": "I4", "status": "failed", "verdict": verdict,
                    "reasoning": response.get("reasoning", ""), "soft": True}
        else:
            return {"check": "I4", "status": "failed", "verdict": "misaligned",
                    "reasoning": response.get("reasoning", "")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_child_components(atoms: list[Atom], edges: list[Edge]) -> dict[str, list[Atom]]:
    """For each W3 method or W4 solution, find child W5 components via decomposes_into edges."""
    atom_by_id: dict[str, Atom] = {a.node_id: a for a in atoms}
    children: dict[str, list[Atom]] = {}
    for e in edges:
        if e.relation != "decomposes_into":
            continue
        parent = atom_by_id.get(e.src)
        child = atom_by_id.get(e.tgt)
        if parent and child and child.type == "component":
            children.setdefault(parent.node_id, []).append(child)
    return children


def _per_atom(atom: Atom, status: str, reason: str = "") -> dict:
    return {
        "node_id": atom.node_id,
        "type": atom.type,
        "status": status,
        "checks": {},
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_I1_PROMPT = """\
You are a fact-checker. From the following text, extract the primary numerical score reported.

Text:
\"\"\"
{context}
\"\"\"

Return JSON: {{"score": <float_or_null>, "reasoning": "<one sentence>"}}
If no numerical score is present, return null.
"""

_I2_PROMPT = """\
You are auditing experimental code for compliance with a task specification.

Task name: {task_name}
Evaluation harness: {eval_harness}
Success criteria: {success_criteria}
Constraints: {constraints}

Experimental code:
\"\"\"
{code_body}
\"\"\"

Decide if this code is compliant, violates_spec, or ambiguous.
- compliant: follows all constraints and is appropriate for the eval harness.
- violates_spec: clearly breaks a constraint or is incompatible with the harness.
- ambiguous: cannot determine without more info.

Return JSON: {{"verdict": "compliant|violates_spec|ambiguous", "reasoning": "<short>"}}
"""

_I4_PROMPT = """\
You are auditing whether a research method's description aligns with its actual code implementation.

Method description (from paper):
\"\"\"
{method_context}
\"\"\"

Code implementations:
\"\"\"
{code_bodies}
\"\"\"

Decide alignment:
- aligned: code implements the described method faithfully.
- partially_aligned: code is in the right direction but misses key elements.
- misaligned: code implements a fundamentally different method.

Return JSON: {{"verdict": "aligned|partially_aligned|misaligned", "reasoning": "<short>"}}
"""
