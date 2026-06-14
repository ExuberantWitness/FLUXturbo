"""NoveltyEvaluator — Hausdorff distance + LLM rubric for research novelty assessment.

Computes layered Hausdorff distances between proposal and existing trajectories,
then uses LLM rubric to analyze divergence points and produce a structured report.
"""

from __future__ import annotations

import numpy as np

from ccchain.core.ontology import Atom, Edge, Trajectory
from ccchain.plugins.base import Evaluator


class NoveltyEvaluator(Evaluator):
    """Evaluate novelty via Hausdorff trajectory distance + LLM rubric."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        hausdorff_weights: dict[str, float] | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.hausdorff_weights = hausdorff_weights or {
            "W2": 0.4,
            "W3": 0.3,
            "W4": 0.2,
            "W5": 0.1,
        }

    def evaluate(
        self,
        proposal_atoms: list[Atom],
        proposal_edges: list[Edge],
        existing_trajectories: list[Trajectory],
    ) -> dict:
        if not existing_trajectories:
            return {
                "novelty_score": 1.0,
                "most_similar_trajectory": None,
                "level_distances": {},
                "divergence_points": [],
                "dimension_scores": {},
                "recommendation": "No existing trajectories — proposal is in uncharted territory.",
            }

        # Build proposal trajectory from atoms
        proposal_traj = self._build_proposal_trajectory(proposal_atoms, proposal_edges)
        proposal_embs = proposal_traj.get_embeddings_by_level()

        # Compute layered Hausdorff distances against each existing trajectory
        best_distance = float("inf")
        best_traj = None
        best_level_dists: dict[str, float] = {}

        for traj in existing_trajectories:
            traj_embs = traj.get_embeddings_by_level()
            level_dists = {}
            total = 0.0

            for lvl in ["W2", "W3", "W4", "W5"]:
                d = _hausdorff(proposal_embs.get(lvl, np.empty((0, 1024))),
                               traj_embs.get(lvl, np.empty((0, 1024))))
                level_dists[lvl] = float(d)
                total += d * self.hausdorff_weights.get(lvl, 0.25)

            if total < best_distance:
                best_distance = total
                best_traj = traj
                best_level_dists = level_dists

        # LLM rubric analysis
        rubric = self._llm_rubric(proposal_traj, best_traj, best_level_dists)

        novelty_score = 1.0 - min(best_distance / 10.0, 1.0)
        novelty_score = max(0.0, min(1.0, novelty_score))

        return {
            "novelty_score": round(novelty_score, 4),
            "most_similar_trajectory": best_traj.to_dict() if best_traj else None,
            "level_distances": best_level_dists,
            "divergence_points": rubric.get("divergence_points", []),
            "dimension_scores": rubric.get("dimension_scores", {}),
            "recommendation": rubric.get("recommendation", ""),
        }

    def _build_proposal_trajectory(
        self, atoms: list[Atom], edges: list[Edge]
    ) -> Trajectory:
        traj = Trajectory()
        for a in atoms:
            if a.level == "W2_problem_analysis":
                traj.W2_problem = a
            elif a.level == "W3_solution_direction":
                traj.W3_solutions.append(a)
            elif a.level == "W4_concrete_solution":
                traj.W4_implementations.append(a)
            elif a.level == "W5_code_implementation":
                traj.W5_code.append(a)
        traj.edges = list(edges)
        return traj

    def _llm_rubric(
        self,
        proposal: Trajectory,
        nearest: Trajectory | None,
        level_dists: dict[str, float],
    ) -> dict:
        from ccchain.core.llm import chat_json

        if nearest is None:
            return {
                "divergence_points": [],
                "dimension_scores": {},
                "recommendation": "No comparison baseline available.",
            }

        prompt = _RUBRIC_PROMPT.format(
            proposal_w2=proposal.W2_problem.context if proposal.W2_problem else "N/A",
            proposal_w3="; ".join(a.context for a in proposal.W3_solutions),
            proposal_w4="; ".join(a.context for a in proposal.W4_implementations),
            nearest_w2=nearest.W2_problem.context if nearest.W2_problem else "N/A",
            nearest_w3="; ".join(a.context for a in nearest.W3_solutions),
            nearest_w4="; ".join(a.context for a in nearest.W4_implementations),
            w2_dist=f"{level_dists.get('W2', 0):.4f}",
            w3_dist=f"{level_dists.get('W3', 0):.4f}",
            w4_dist=f"{level_dists.get('W4', 0):.4f}",
            w5_dist=f"{level_dists.get('W5', 0):.4f}",
        )

        return chat_json(
            [{"role": "user", "content": prompt}],
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Hausdorff distance between two embedding sets.

    H(A,B) = max( sup_{x∈A} inf_{y∈B} ||x-y||, sup_{y∈B} inf_{x∈A} ||x-y|| )
    """
    if a.size == 0 or b.size == 0:
        return 1.0 if (a.size > 0 or b.size > 0) else 0.0

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    # Pairwise distances: (|A|, |B|)
    from scipy.spatial.distance import cdist

    dists = cdist(a, b, metric="cosine")

    a_to_b = dists.min(axis=1).max()  # sup inf
    b_to_a = dists.min(axis=0).max()  # sup inf
    return float(max(a_to_b, b_to_a))


_RUBRIC_PROMPT = """\
You are a research novelty evaluator. Compare a proposal against the nearest existing trajectory.

PROPOSAL:
  W2 (Problem): {proposal_w2}
  W3 (Directions): {proposal_w3}
  W4 (Solutions): {proposal_w4}

NEAREST EXISTING TRAJECTORY:
  W2 (Problem): {nearest_w2}
  W3 (Directions): {nearest_w3}
  W4 (Solutions): {nearest_w4}

LAYERED HAUSDORFF DISTANCES (cosine):
  W2: {w2_dist}
  W3: {w3_dist}
  W4: {w4_dist}
  W5: {w5_dist}

Analyze divergence points across these dimensions:
- problem_formulation (is the problem framed differently?)
- solution_approach (are the methods fundamentally different?)
- technical_mechanism (are the core mechanisms novel?)
- evaluation_perspective (does it enable new types of evaluation?)

Return JSON:
{{
  "divergence_points": ["specific point 1", "specific point 2"],
  "dimension_scores": {{
    "problem_formulation": 0.7,
    "solution_approach": 0.8,
    "technical_mechanism": 0.6,
    "evaluation_perspective": 0.5
  }},
  "recommendation": "1-2 sentence assessment of novelty and whether to pursue."
}}
"""
