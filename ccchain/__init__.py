"""ccchain — Claim Chain knowledge base for AI autonomous research.

Three public methods:
    ingest(segments, source_pdf) → (result, error)
    search(query, top_k, level) → (result, error)
    evaluate(proposal_text, domain) → (result, error)
"""

from __future__ import annotations

from ccchain.config import Config
from ccchain.core.store import CCStore
from ccchain.plugins.base import (
    Evaluator,
    Extractor,
    Reducer,
    Refiner,
    Retriever,
)
from ccchain.plugins.extraction import TwoPhaseExtractor
from ccchain.plugins.refinement import LeapRefiner
from ccchain.plugins.reduction import HierarchicalReducer
from ccchain.plugins.retrieval import GraphRetriever
from ccchain.plugins.evaluation import NoveltyEvaluator

# Lazy singletons
_store: CCStore | None = None
_extractor: Extractor | None = None
_refiner: Refiner | None = None
_reducer: Reducer | None = None
_retriever: Retriever | None = None
_evaluator: Evaluator | None = None


def _init(config: Config | None = None):
    global _store, _extractor, _refiner, _reducer, _retriever, _evaluator

    if config is None:
        config = Config()

    if _store is None:
        _store = CCStore(config.db_path, config.graph_dir)

    if _extractor is None:
        _extractor = TwoPhaseExtractor(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
        )

    if _refiner is None:
        _refiner = LeapRefiner(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
        )

    if _reducer is None:
        _reducer = HierarchicalReducer(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
        )

    if _retriever is None:
        _retriever = GraphRetriever(
            embedder_base_url=config.embedder_base_url,
            embedder_model=config.embedder_model,
            embedder_api_key=config.llm_api_key,
            link_top_k=config.link_top_k,
            ppr_damping=config.ppr_damping,
            w5_bias_weight=config.w5_bias_weight,
        )

    if _evaluator is None:
        _evaluator = NoveltyEvaluator(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            hausdorff_weights=config.hausdorff_weights,
        )


def ingest(
    segments: list[str],
    source_pdf: str,
    version: int = 1,
    on_progress: "callable | None" = None,
) -> tuple[dict | None, str | None]:
    """Ingest paper text segments and build the CC knowledge index.

    Pipeline: extract (two-phase) → refine (iterative) → store (dual-write)
              → reduce (incremental per connected component)

    Args:
        segments: List of text segments from the paper.
        source_pdf: Source PDF filename for provenance.
        version: Blueprint version number.
        on_progress: Optional callback(stage: str, pct: float) for progress.

    Returns:
        (result, error) tuple. On success, error=None and result contains
        {node_count_by_level, edge_count, trajectory_count}.
    """
    try:
        _init()
        from ccchain.core.gatekeeper import validate
        from ccchain.core.embedding import embed

        config = Config()

        def _progress(stage: str, pct: float):
            if on_progress:
                on_progress(stage, pct)

        # 1. Extract
        _progress("extracting", 0.0)
        atoms, edges = _extractor.extract(segments, source_pdf)
        _progress("extracting", 0.5)

        # Embed all atoms
        if atoms:
            texts = [a.context for a in atoms]
            embeddings = embed(
                texts,
                base_url=config.embedder_base_url,
                model=config.embedder_model,
                api_key=config.llm_api_key,
            )
            for a, emb in zip(atoms, embeddings):
                a.embedding = emb
        _progress("extracting", 1.0)

        # 2. Refine
        _progress("refining", 0.0)
        atoms, edges, fix_log = _refiner.refine(
            atoms, edges, segments, max_rounds=config.max_refine_rounds
        )
        _progress("refining", 1.0)

        # 3. Store
        _progress("storing", 0.0)
        result = _store.insert_blueprint(atoms, edges, source_pdf)
        _progress("storing", 1.0)

        # 4. Reduce (incremental per connected component)
        _progress("reducing", 0.0)
        for from_level, to_level in [
            ("W5_code_implementation", "W4_concrete_solution"),
            ("W4_concrete_solution", "W3_solution_direction"),
            ("W3_solution_direction", "W2_problem_analysis"),
        ]:
            new_atoms = _reducer.reduce_level(
                atoms, edges, from_level, to_level, _store.graph
            )
            if new_atoms:
                # Embed new atoms
                texts = [a.context for a in new_atoms]
                embeddings = embed(
                    texts,
                    base_url=config.embedder_base_url,
                    model=config.embedder_model,
                    api_key=config.llm_api_key,
                )
                for a, emb in zip(new_atoms, embeddings):
                    a.embedding = emb

                _store.upsert_atoms(new_atoms)

                # Create AGGREGATES_TO edges from source atoms to reduced atoms
                from ccchain.core.ontology import Edge as OntEdge

                reduced_ids = {a.node_id for a in new_atoms}
                agg_edges: list[OntEdge] = []
                for a in atoms:
                    if a.level == from_level:
                        for rid in reduced_ids:
                            agg_edges.append(OntEdge(
                                src=a.node_id,
                                relation="aggregates_to",
                                tgt=rid,
                            ))
                _store.upsert_edges(agg_edges)

        _progress("reducing", 1.0)

        # Build result summary
        trajectories = _store.get_all_trajectories()
        level_counts: dict[str, int] = {}
        for lvl in ["W2_problem_analysis", "W3_solution_direction",
                     "W4_concrete_solution", "W5_code_implementation"]:
            level_counts[lvl] = len(_store.query_by_level(lvl))

        _progress("done", 1.0)

        return {
            "node_count_by_level": level_counts,
            "edge_count": result["inserted_edges"],
            "trajectory_count": len(trajectories),
        }, None

    except Exception as exc:
        return None, str(exc)


def search(
    query: str,
    top_k: int = 10,
    level: str = "W4",
) -> tuple[list[dict] | None, str | None]:
    """Search the CC knowledge base.

    Pipeline: dual embed(query) → dense recall → PPR diffusion → filter by level.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        level: Level filter (W2/W3/W4/W5 or full level name).

    Returns:
        (result, error) tuple.
    """
    try:
        _init()

        embeddings = _store.get_all_embeddings()
        if embeddings.shape[0] == 0:
            return [], None

        results = _retriever.search(
            query=query,
            top_k=top_k,
            level=level,
            graph=_store.graph,
            embeddings=embeddings,
            node_index=_store._node_index,
        )

        # Enrich results with context
        for r in results:
            atom = _store.query_by_id(r["node_id"])
            if atom:
                r["context"] = atom.context
                r["source_pdf"] = atom.source_pdf

        return results, None

    except Exception as exc:
        return None, str(exc)


def evaluate(
    proposal_text: str,
    domain: str | None = None,
    on_progress: "callable | None" = None,
) -> tuple[dict | None, str | None]:
    """Evaluate the novelty of a research proposal.

    Temporarily integrates the proposal into the graph (status='transient'),
    builds its trajectory, computes Hausdorff distances against existing
    trajectories, runs LLM rubric analysis, then deletes all temporary data.

    Args:
        proposal_text: Description of the research proposal/idea.
        domain: Optional domain filter for comparison trajectories.
        on_progress: Optional callback(stage: str, pct: float).

    Returns:
        (result, error) tuple.
    """
    try:
        _init()
        from ccchain.core.embedding import embed
        config = Config()

        def _progress(stage: str, pct: float):
            if on_progress:
                on_progress(stage, pct)

        # 1. Extract proposal
        _progress("extracting", 0.0)
        atoms, edges = _extractor.extract([proposal_text], "proposal")
        for a in atoms:
            a.status = "transient"
        _progress("extracting", 0.5)

        # 2. Embed
        if atoms:
            texts = [a.context for a in atoms]
            embeddings = embed(
                texts,
                base_url=config.embedder_base_url,
                model=config.embedder_model,
                api_key=config.llm_api_key,
            )
            for a, emb in zip(atoms, embeddings):
                a.embedding = emb
        _progress("extracting", 1.0)

        # 3. Store temporarily
        _progress("integrating", 0.0)
        _store.insert_blueprint(atoms, edges, "proposal")
        _progress("integrating", 1.0)

        # 4. Get existing trajectories for comparison
        _progress("evaluating", 0.0)
        existing = _store.get_all_trajectories(domain)
        _progress("evaluating", 0.3)

        # 5. Run evaluation
        report = _evaluator.evaluate(atoms, edges, existing)
        _progress("evaluating", 0.8)

        # 6. Cleanup transient data
        deleted = _store.delete_transient()
        report["_deleted_transient"] = deleted
        _progress("evaluating", 1.0)

        _progress("done", 1.0)
        return report, None

    except Exception as exc:
        # Best-effort cleanup
        try:
            if _store is not None:
                _store.delete_transient()
        except Exception:
            pass
        return None, str(exc)
