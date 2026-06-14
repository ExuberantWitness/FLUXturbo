"""Plugin abstract base classes — 5 single-method ABCs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import igraph as ig
import numpy as np

from ccchain.core.ontology import Atom, Edge, Trajectory


class Extractor(ABC):
    """Extract atoms and edges from raw text segments."""

    @abstractmethod
    def extract(
        self,
        segments: list[str],
        source_pdf: str,
    ) -> tuple[list[Atom], list[Edge]]:
        """Two-phase extraction: W2+W3 first, then W4+W5 conditioned on Phase 1."""
        ...


class Refiner(ABC):
    """Validate and iteratively fix extraction results."""

    @abstractmethod
    def refine(
        self,
        atoms: list[Atom],
        edges: list[Edge],
        segments: list[str] | None = None,
        max_rounds: int = 3,
    ) -> tuple[list[Atom], list[Edge], dict]:
        """Run gatekeeper → feedback LLM to fix → re-validate up to max_rounds.

        Returns (final_atoms, final_edges, fix_log).
        """
        ...


class Reducer(ABC):
    """Semantically reduce lower-level atoms into higher-level abstractions."""

    @abstractmethod
    def reduce_level(
        self,
        atoms: list[Atom],
        edges: list[Edge],
        from_level: str,
        to_level: str,
        graph: ig.Graph,
    ) -> list[Atom]:
        """Group same-level atoms by connected component → LLM semantic induction.

        Returns newly created higher-level atoms.
        """
        ...


class Retriever(ABC):
    """HippoRAG-style retrieval: dual embedding → dense recall → PPR diffusion."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        level: str,
        graph: ig.Graph,
        embeddings: np.ndarray,
        node_index: dict[str, int],
    ) -> list[dict]:
        """Retrieve top_k atoms at the given level for the query.

        node_index maps atom node_id → igraph vertex index.
        """
        ...


class Evaluator(ABC):
    """Evaluate novelty of a research proposal against existing trajectories."""

    @abstractmethod
    def evaluate(
        self,
        proposal_atoms: list[Atom],
        proposal_edges: list[Edge],
        existing_trajectories: list[Trajectory],
    ) -> dict:
        """Hausdorff distance + LLM rubric → structured novelty report."""
        ...
