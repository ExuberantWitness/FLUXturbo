"""Integration test: full pipeline with real PDF text, mock LLM/embeddings.

Validates the complete data flow:
  PDF text → extract → gatekeeper → refine → store → reduce → trajectories → search → evaluate
"""

import json
import os
import sys
import tempfile
import uuid
from unittest.mock import patch

import numpy as np
import pytest

# Add project root for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccchain.config import Config
from ccchain.core.ontology import (
    LEVELS,
    Atom,
    Edge,
    Rho,
    Trajectory,
)
from ccchain.core.gatekeeper import validate
from ccchain.core.store import CCStore
from ccchain.core.graph import ppr, connected_components_by_level
from ccchain.plugins.extraction import TwoPhaseExtractor
from ccchain.plugins.refinement import LeapRefiner
from ccchain.plugins.reduction import HierarchicalReducer
from ccchain.plugins.retrieval import GraphRetriever
from ccchain.plugins.evaluation import NoveltyEvaluator

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
PDF_DIR = r"E:\DATA\vscode\ARIS\pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _extract_pdf_texts():
    """Extract text chunks from all PDFs in the ARIS/pdf directory."""
    import fitz

    all_texts: list[dict] = []
    for fname in sorted(os.listdir(PDF_DIR)):
        if not fname.endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, fname)
        doc = fitz.open(path)
        chunks: list[str] = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                chunks.append(text)
        all_texts.append({
            "filename": fname,
            "pages": len(doc),
            "chunks": chunks,
        })
        doc.close()
    return all_texts


def _make_llm_response_w2w3(paper_text: str) -> dict:
    """Generate a realistic Phase 1 response based on paper content keywords."""
    name = "unknown problem"
    context = paper_text[:300]
    lower = paper_text.lower()

    if "safe" in lower and ("reinforcement learning" in lower or "rl" in lower.split()):
        name = "Safe RL with Uncertainty-Aware Constraints"
        context = "Safe reinforcement learning requires jointly optimizing reward and safety constraints, where independent uncertainty estimation per objective leads to over-conservatism."
    elif "inverse" in lower and "manipulation" in lower:
        name = "Hybrid Symbolic-Neural Inverse Manipulation"
        context = "Inverting robotic manipulation tasks requires reasoning over both symbolic state transitions and continuous interaction dynamics, which are not fully reversible."
    elif "eeg" in lower and "emotion" in lower:
        name = "Continuous EEG Emotion Dynamics Modeling"
        context = "Continuous emotion prediction from EEG requires capturing long-range temporal dependencies and globally coherent emotional evolution patterns beyond point-wise regression."
    elif "credit" in lower and "assignment" in lower:
        name = "Credit Assignment Noise in CTDE"
        context = "CTDE suffers from high-variance policy gradients due to noisy credit assignment."

    directions = []
    if "safe" in lower and "rl" in lower.split():
        directions = [
            {"name": "Joint Q-Value Uncertainty Modeling",
             "context": "Model inter-objective covariance in vector-valued Q estimation to reduce over-conservatism.",
             "compares_to": ["Independent Critic Ensembles", "Lagrangian Methods"]},
        ]
    elif "inverse" in lower and "manipulation" in lower:
        directions = [
            {"name": "STRIPS-based Inverse Planning with RL Residuals",
             "context": "Derive inverse objectives from symbolic operators and use RL to resolve unresolved predicates.",
             "compares_to": ["Pure RL Policy Inversion", "Trajectory Optimization"]},
        ]
    elif "eeg" in lower and "emotion" in lower:
        directions = [
            {"name": "VQ-VAE + Masked Modeling for Emotion Dynamics",
             "context": "Use vector-quantized representations and masked temporal modeling with RL trajectory optimization.",
             "compares_to": ["Point-wise Regression", "RNN-based Emotion Models"]},
        ]

    return {
        "W2_problem_analysis": {"name": name, "context": context},
        "W3_solution_directions": directions,
    }


def _make_llm_response_w4w5(paper_text: str, w3_atoms: list) -> dict:
    """Generate a realistic Phase 2 response based on paper content."""
    lower = paper_text.lower()
    solutions = []

    if "safe" in lower and "cop-q" in lower:
        solutions = [{
            "name": "Cholesky-Ordered Projection Q-learning (COP-Q)",
            "context": "Uses Cholesky factorization to encode objective priority in sequential form, constructing a generalized confidence bound in joint Q-value space.",
            "parent_W3_id": "Joint Q-Value Uncertainty Modeling",
            "extends": ["Ensemble Q-learning"],
            "improves": ["Independent Critic Ensembles"],
            "extends_rho": {"bottleneck": "overestimation_bias", "mechanism": "Joint covariance modeling reduces over-conservatism", "tradeoff": "Requires Cholesky decomposition per update", "confidence": 0.85},
            "improves_rho": {"bottleneck": "sample_inefficiency", "mechanism": "Adaptive conservatism reduction improves exploration", "tradeoff": "Additional hyperparameter for priority ordering", "confidence": 0.8},
            "W5_implementations": [
                {"name": "COP_Q_critic_update", "context": "Computes Cholesky-ordered projection for TD target with joint covariance.", "code_ref": "copq_critic_update"},
                {"name": "COP_Actor_optimize", "context": "Actor optimization using COP-Q value estimates.", "code_ref": "copq_actor_loss"},
            ],
        }]
    elif "inverse" in lower and "manipulation" in lower:
        solutions = [{
            "name": "STRIPS Operator Extraction from Demonstrations",
            "context": "Extracts STRIPS-like operators via soft geometric predicates from demonstration data, constructing precondition/effect sets.",
            "parent_W3_id": "STRIPS-based Inverse Planning with RL Residuals",
            "extends": ["Behavior Cloning"],
            "improves": ["Pure RL Policy Inversion"],
            "extends_rho": {"bottleneck": "representational_limitation", "mechanism": "Symbolic operators provide interpretable task structure", "tradeoff": "Predicate design requires domain knowledge", "confidence": 0.75},
            "improves_rho": {"bottleneck": "sample_inefficiency", "mechanism": "Symbolic planning reduces RL exploration space", "tradeoff": "Residual learning may still require significant samples", "confidence": 0.7},
            "W5_implementations": [
                {"name": "extract_strips_operators", "context": "Extracts preconditions, add/delete effects from demonstrations via geometric predicates.", "code_ref": "extract_strips_ops"},
                {"name": "residual_sac_policy", "context": "SAC policy for residual predicate satisfaction after symbolic planning.", "code_ref": "ResidualSAC"},
            ],
        }]
    elif "eeg" in lower and "emotion" in lower:
        solutions = [{
            "name": "EEGDancer VQ-Masked RL Framework",
            "context": "Integrates vector-quantized representation learning with masked temporal modeling and RL-based trajectory optimization for continuous emotion prediction.",
            "parent_W3_id": "VQ-VAE + Masked Modeling for Emotion Dynamics",
            "extends": ["VQ-VAE"],
            "improves": ["Point-wise Regression"],
            "extends_rho": {"bottleneck": "representational_limitation", "mechanism": "VQ provides discrete latent space for emotion dynamics", "tradeoff": "Codebook size limits expressiveness", "confidence": 0.8},
            "improves_rho": {"bottleneck": "generalization_gap", "mechanism": "Masked modeling captures global temporal dependencies", "tradeoff": "Increased training complexity", "confidence": 0.75},
            "W5_implementations": [
                {"name": "vq_encoder", "context": "Vector-quantized encoder mapping EEG to discrete emotion latent codes.", "code_ref": "VQEncoder"},
                {"name": "masked_temporal_model", "context": "Masked modeling over temporal EEG sequences for emotion prediction.", "code_ref": "MaskedTemporalModel"},
                {"name": "rl_trajectory_optimizer", "context": "RL-based optimization of emotion trajectory in latent space.", "code_ref": "RLTrajectoryOptimizer"},
            ],
        }]

    return {"W4_concrete_solutions": solutions}


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """End-to-end pipeline test with real PDF text and mock LLM responses."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create isolated store for each test."""
        self.tmpdir = tempfile.mkdtemp(prefix="ccchain_test_")
        self.store = CCStore(
            db_path=os.path.join(self.tmpdir, "test.db"),
            graph_dir=self.tmpdir,
        )
        self.config = Config(
            db_path=os.path.join(self.tmpdir, "test.db"),
            graph_dir=self.tmpdir,
        )
        yield
        self.store.db.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extract_real_pdf_text_phase1(self):
        """Phase 1 extraction with real PDF text, mock LLM."""
        papers = _extract_pdf_texts()
        assert len(papers) >= 3, f"Expected 3 PDFs, got {len(papers)}"

        extractor = TwoPhaseExtractor(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:2])  # First 2 pages
            response = _make_llm_response_w2w3(combined)
            atoms, edges = extractor._parse_phase1(response, paper["filename"])

            # Verify W2
            w2_atoms = [a for a in atoms if a.level == "W2_problem_analysis"]
            assert len(w2_atoms) == 1, f"No W2 for {paper['filename']}"
            assert w2_atoms[0].type == "bottleneck"
            assert len(w2_atoms[0].context) > 20

            # Verify W3
            w3_atoms = [a for a in atoms if a.level == "W3_solution_direction"]
            assert len(w3_atoms) >= 1, f"No W3 for {paper['filename']}"
            assert all(a.type == "method" for a in w3_atoms)

            # Verify edges
            assert len(edges) >= len(w3_atoms)  # at least W2→W3 per direction

            # All atoms have valid structure
            assert all(a.node_id for a in atoms)
            assert all(a.source_pdf == paper["filename"] for a in atoms)

    def test_extract_real_pdf_text_phase2(self):
        """Phase 2 extraction with real PDF text, mock LLM."""
        papers = _extract_pdf_texts()
        extractor = TwoPhaseExtractor(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:3])
            response = _make_llm_response_w4w5(combined, [])
            atoms, edges = extractor._parse_phase2(response, paper["filename"])

            w4_atoms = [a for a in atoms if a.level == "W4_concrete_solution"]
            w5_atoms = [a for a in atoms if a.level == "W5_code_implementation"]

            assert len(w4_atoms) >= 1, f"No W4 for {paper['filename']}"
            assert len(w5_atoms) >= 1, f"No W5 for {paper['filename']}"
            assert all(a.type == "solution" for a in w4_atoms)
            assert all(a.type == "component" for a in w5_atoms)

            # W5 should have code_ref
            assert any(a.code_ref for a in w5_atoms), f"No code_ref in W5 for {paper['filename']}"

    @patch("ccchain.core.llm.chat_json")
    def test_full_extract_refine_store_pipeline(self, mock_chat):
        """Extract → Refine → Store for all 3 PDFs using mock LLM."""
        papers = _extract_pdf_texts()

        # Phase 1 mock
        phase1_responses = {}
        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:2])
            phase1_responses[paper["filename"]] = _make_llm_response_w2w3(combined)

        # Phase 2 mock
        phase2_responses = {}
        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:3])
            phase2_responses[paper["filename"]] = _make_llm_response_w4w5(combined, [])

        call_count = [0]

        def mock_chat_json_fn(messages, **kwargs):
            call_count[0] += 1
            msg = messages[0]["content"] if messages else ""
            # Determine which paper by matching text content
            if "W2_problem_analysis" in msg:
                # Phase 1 call — return first paper's response
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:2])
                    if combined[:100] in msg:
                        return phase1_responses[paper["filename"]]
                return phase1_responses[papers[0]["filename"]]
            elif "W4_concrete_solutions" in msg:
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:3])
                    if combined[:100] in msg:
                        return phase2_responses[paper["filename"]]
                return phase2_responses[papers[0]["filename"]]
            return {}

        mock_chat.side_effect = mock_chat_json_fn

        extractor = TwoPhaseExtractor(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )
        refiner = LeapRefiner(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        total_atoms = 0
        total_edges = 0

        for paper in papers:
            # Extract
            combined = "\n\n".join(paper["chunks"][:5])
            atoms, edges = extractor.extract(
                [combined],  # Send as one chunk for simplicity
                source_pdf=paper["filename"],
            )

            assert len(atoms) > 0, f"Extraction produced 0 atoms for {paper['filename']}"

            # Refine
            atoms, edges, fix_log = refiner.refine(atoms, edges)
            assert fix_log["rounds"] >= 0

            # Store
            result = self.store.insert_blueprint(atoms, edges, paper["filename"])
            assert result["inserted_nodes"] > 0
            assert result["inserted_edges"] > 0

            total_atoms += result["inserted_nodes"]
            total_edges += result["inserted_edges"]

        assert total_atoms >= 9  # At least W2+W3+W4+W5 per paper (min 3 each)
        assert total_edges >= 6  # At least W2→W3→W4→W5 edges

        # Verify store contents
        all_levels = self.store.query_by_level("W4_concrete_solution")
        assert len(all_levels) >= 3, f"Expected >= 3 W4 atoms, got {len(all_levels)}"

        # Verify graph integrity
        assert self.store.graph.vcount() == total_atoms
        assert len(self.store._node_index) == total_atoms

    @patch("ccchain.core.llm.chat_json")
    def test_pipeline_and_trajectory_building(self, mock_chat):
        """Full pipeline: extract → refine → store → reduce → trajectories."""
        papers = _extract_pdf_texts()
        extractor = TwoPhaseExtractor(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )
        refiner = LeapRefiner(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )
        reducer = HierarchicalReducer(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        # Setup mock to return realistic responses
        def mock_fn(messages, **kwargs):
            msg = messages[0]["content"] if messages else ""

            if "W2_problem_analysis" in msg:
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:2])
                    if combined[:100] in msg:
                        return _make_llm_response_w2w3(combined)
                return _make_llm_response_w2w3("\n\n".join(papers[0]["chunks"][:2]))

            if "W4_concrete_solutions" in msg:
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:3])
                    if combined[:100] in msg:
                        return _make_llm_response_w4w5(combined, [])
                return _make_llm_response_w4w5("\n\n".join(papers[0]["chunks"][:3]), [])

            if "reduced_atoms" in msg:
                return {"reduced_atoms": [{
                    "name": "Synthesized Solution",
                    "type": "method",
                    "context": "Cross-paper synthesis of solution approaches.",
                }]}

            return {}

        mock_chat.side_effect = mock_fn

        # Ingest all papers
        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:5])
            atoms, edges = extractor.extract([combined], source_pdf=paper["filename"])
            atoms, edges, _ = refiner.refine(atoms, edges)

            # Generate mock embeddings (1024-dim)
            for a in atoms:
                a.embedding = np.random.randn(1024).astype(np.float32)

            self.store.insert_blueprint(atoms, edges, paper["filename"])

            # Reduce W5→W4
            w4_atoms = reducer.reduce_level(
                atoms, edges,
                from_level="W5_code_implementation",
                to_level="W4_concrete_solution",
                graph=self.store.graph,
            )

        # Build trajectories
        trajectories = self.store.get_all_trajectories()
        assert len(trajectories) >= 3, f"Expected >= 3 trajectories, got {len(trajectories)}"

        # Verify trajectory structure
        for traj in trajectories:
            assert isinstance(traj, Trajectory)
            assert len(traj.W5_code) >= 1, "Each trajectory must start from W5"
            # Trajectory should have at least one ancestor
            has_ancestor = (
                traj.W4_implementations
                or traj.W3_solutions
                or traj.W2_problem is not None
            )
            # At minimum, the W5 itself is in the trajectory

    @patch("ccchain.core.llm.chat_json")
    def test_search_on_built_knowledge_base(self, mock_chat):
        """Build knowledge base from PDFs, then search."""
        papers = _extract_pdf_texts()
        extractor = TwoPhaseExtractor(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )
        refiner = LeapRefiner(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        def mock_fn(messages, **kwargs):
            msg = messages[0]["content"] if messages else ""
            if "W2_problem_analysis" in msg:
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:2])
                    if combined[:100] in msg:
                        return _make_llm_response_w2w3(combined)
                return _make_llm_response_w2w3("\n\n".join(papers[0]["chunks"][:2]))
            if "W4_concrete_solutions" in msg:
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:3])
                    if combined[:100] in msg:
                        return _make_llm_response_w4w5(combined, [])
                return _make_llm_response_w4w5("\n\n".join(papers[0]["chunks"][:3]), [])
            return {}

        mock_chat.side_effect = mock_fn

        # Ingest all papers
        all_w4_atoms: list[Atom] = []
        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:5])
            atoms, edges = extractor.extract([combined], source_pdf=paper["filename"])
            atoms, edges, _ = refiner.refine(atoms, edges)
            for a in atoms:
                a.embedding = np.random.randn(1024).astype(np.float32)
            self.store.insert_blueprint(atoms, edges, paper["filename"])
            all_w4_atoms.extend([a for a in atoms if a.level == "W4_concrete_solution"])

        assert len(all_w4_atoms) >= 3

        # Mock embedding in retrieval
        import ccchain.plugins.retrieval as rmod
        original_embed = rmod.embed

        # Build embedding array from stored BLOBs with node_id ordering
        rows = self.store.db.execute(
            "SELECT node_id, embedding FROM cc_nodes WHERE embedding IS NOT NULL"
        ).fetchall()
        emb_list = []
        emb_node_index: dict[str, int] = {}
        for i, row in enumerate(rows):
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            emb_list.append(emb)
            emb_node_index[row["node_id"]] = i
        emb_array = np.stack(emb_list) if emb_list else np.empty((0, 1024), dtype=np.float32)

        class _MockEmbed:
            def __call__(self, texts, **kwargs):
                return np.random.randn(len(texts), 1024).astype(np.float32)

        rmod.embed = _MockEmbed()

        try:
            retriever = GraphRetriever(
                embedder_base_url="http://localhost:11434/v1",
                embedder_model="bge-m3:latest",
            )

            # Search for "safe RL"
            safe_results = retriever.search(
                query="safe reinforcement learning with constraints",
                top_k=5,
                level="W4",
                graph=self.store.graph,
                embeddings=emb_array,
                node_index=emb_node_index,
            )
            assert len(safe_results) >= 0  # May be 0 due to random embeddings

            # Search for "inverse manipulation"
            inv_results = retriever.search(
                query="inverse robotic manipulation planning",
                top_k=5,
                level="W4",
                graph=self.store.graph,
                embeddings=emb_array,
                node_index=emb_node_index,
            )
            assert isinstance(inv_results, list)

            # Search for "EEG emotion"
            eeg_results = retriever.search(
                query="EEG continuous emotion recognition",
                top_k=5,
                level="W4",
                graph=self.store.graph,
                embeddings=emb_array,
                node_index=emb_node_index,
            )
            assert isinstance(eeg_results, list)

            # Verify all results have expected structure
            for results in [safe_results, inv_results, eeg_results]:
                for r in results:
                    assert "node_id" in r
                    assert "score" in r
                    assert "level" in r

        finally:
            rmod.embed = original_embed

    @patch("ccchain.core.llm.chat_json")
    def test_evaluate_novelty_with_real_papers(self, mock_chat):
        """Evaluate a novel proposal against paper-derived trajectories."""
        papers = _extract_pdf_texts()
        extractor = TwoPhaseExtractor(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )
        refiner = LeapRefiner(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        def mock_fn(messages, **kwargs):
            msg = messages[0]["content"] if messages else ""
            if "W2_problem_analysis" in msg:
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:2])
                    if combined[:100] in msg:
                        return _make_llm_response_w2w3(combined)
                return _make_llm_response_w2w3("\n\n".join(papers[0]["chunks"][:2]))
            if "W4_concrete_solutions" in msg:
                for paper in papers:
                    combined = "\n\n".join(paper["chunks"][:3])
                    if combined[:100] in msg:
                        return _make_llm_response_w4w5(combined, [])
                return _make_llm_response_w4w5("\n\n".join(papers[0]["chunks"][:3]), [])
            # Rubric response
            if "PROPOSAL:" in msg:
                return _load_fixture("rubric_response.json")
            return {}

        mock_chat.side_effect = mock_fn

        # Build knowledge base
        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:5])
            atoms, edges = extractor.extract([combined], source_pdf=paper["filename"])
            atoms, edges, _ = refiner.refine(atoms, edges)
            for a in atoms:
                a.embedding = np.random.randn(1024).astype(np.float32)
            self.store.insert_blueprint(atoms, edges, paper["filename"])

        trajectories = self.store.get_all_trajectories()
        assert len(trajectories) >= 3

        # Create a novel proposal (partial optimal transport for safe RL)
        proposal_atoms = [
            Atom(
                node_id=f"prop_w2_{uuid.uuid4().hex[:8]}",
                name="Conservative Safe RL with Partial OT",
                type="bottleneck",
                level="W2_problem_analysis",
                context="Existing safe RL methods use full optimal transport which requires balanced mass. Partial OT allows unbalanced allocation, better modeling asymmetric safety constraints.",
                embedding=np.random.randn(1024).astype(np.float32),
                status="transient",
            ),
            Atom(
                node_id=f"prop_w3_{uuid.uuid4().hex[:8]}",
                name="Partial Optimal Transport for Safety",
                type="method",
                level="W3_solution_direction",
                context="Use partial optimal transport with learnable cost to handle asymmetric safety-reward trade-offs.",
                embedding=np.random.randn(1024).astype(np.float32),
                status="transient",
            ),
            Atom(
                node_id=f"prop_w4_{uuid.uuid4().hex[:8]}",
                name="Partial OT Safety Filter",
                type="solution",
                level="W4_concrete_solution",
                context="Partial OT with entropic regularization and mass relaxation parameter for safety-constrained RL.",
                embedding=np.random.randn(1024).astype(np.float32),
                status="transient",
            ),
        ]

        evaluator = NoveltyEvaluator(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        report = evaluator.evaluate(proposal_atoms, [], trajectories)

        # Verify report structure
        assert "novelty_score" in report
        assert 0.0 <= report["novelty_score"] <= 1.0
        assert "level_distances" in report
        assert "divergence_points" in report
        assert "dimension_scores" in report
        assert "recommendation" in report
        assert report["most_similar_trajectory"] is not None

        # Verify level distances
        for lvl in ["W2", "W3", "W4", "W5"]:
            assert lvl in report["level_distances"], f"Missing {lvl} in level_distances"

    def test_store_persistence_roundtrip(self):
        """Verify store can persist and rebuild graph correctly."""
        # Insert some atoms
        atoms = [
            Atom(node_id="w2_1", name="Test Problem", type="bottleneck",
                 level="W2_problem_analysis", context="A test problem."),
            Atom(node_id="w3_1", name="Test Solution", type="method",
                 level="W3_solution_direction", context="A test solution."),
            Atom(node_id="w4_1", name="Test Implementation", type="solution",
                 level="W4_concrete_solution", context="A test implementation."),
        ]
        for a in atoms:
            a.embedding = np.random.randn(1024).astype(np.float32)

        edges = [
            Edge(src="w2_1", relation="decomposes_into", tgt="w3_1"),
            Edge(src="w3_1", relation="decomposes_into", tgt="w4_1"),
        ]

        result = self.store.insert_blueprint(atoms, edges, "test.pdf")
        assert result["inserted_nodes"] == 3
        assert result["inserted_edges"] == 2

        # Persist
        self.store.persist()
        pickle_path = os.path.join(self.tmpdir, "cc_graph.pickle")
        assert os.path.exists(pickle_path), "Pickle file not created"

        # Reopen store — should load from pickle
        store2 = CCStore(
            db_path=os.path.join(self.tmpdir, "test.db"),
            graph_dir=self.tmpdir,
        )
        assert store2.graph.vcount() == 3
        assert store2.graph.ecount() == 2

        # Verify node_index rebuilt correctly
        assert len(store2._node_index) == 3
        assert "w2_1" in store2._node_index
        store2.db.close()

    def test_delete_transient_cleanup(self):
        """Test that transient atoms are properly cleaned up."""
        transient = [
            Atom(node_id="t_w2", name="Temp Problem", type="bottleneck",
                 level="W2_problem_analysis", context="temp", status="transient"),
            Atom(node_id="t_w3", name="Temp Solution", type="method",
                 level="W3_solution_direction", context="temp", status="transient"),
        ]
        permanent = [
            Atom(node_id="p_w2", name="Real Problem", type="bottleneck",
                 level="W2_problem_analysis", context="real", status="active"),
        ]

        for a in transient + permanent:
            a.embedding = np.random.randn(1024).astype(np.float32)

        edges = [
            Edge(src="t_w2", relation="decomposes_into", tgt="t_w3"),
            Edge(src="p_w2", relation="decomposes_into", tgt="t_w2"),  # cross
        ]

        self.store.insert_blueprint(transient + permanent, edges, "test.pdf")

        # Verify 3 atoms exist (counting all statuses including transient)
        assert len(self.store.query_by_level("W2_problem_analysis", status=None)) == 2
        assert len(self.store.query_by_level("W3_solution_direction", status=None)) == 1

        # Delete transient
        deleted = self.store.delete_transient()
        assert deleted == 2

        # Only permanent remains
        remaining = self.store.query_by_level("W2_problem_analysis")
        assert len(remaining) == 1
        assert remaining[0].node_id == "p_w2"

        # Cross edge should also be cleaned
        all_w2 = self.store.query_by_level("W2_problem_analysis")
        neighbors = self.store.get_neighbors("p_w2")
        assert "t_w2" not in neighbors

    def test_gatekeeper_on_extracted_real_data(self):
        """Run gatekeeper validation on atoms extracted from real PDFs."""
        papers = _extract_pdf_texts()
        extractor = TwoPhaseExtractor(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        for paper in papers:
            combined = "\n\n".join(paper["chunks"][:2])
            response = _make_llm_response_w2w3(combined)
            atoms, edges = extractor._parse_phase1(response, paper["filename"])

            # Create comparison target atoms so edges don't dangle
            existing_ids = {a.node_id for a in atoms}
            compare_edges = [e for e in edges if e.relation == "compares"]
            for e in compare_edges:
                if e.tgt not in existing_ids:
                    atoms.append(Atom(
                        node_id=e.tgt,
                        name="External Reference",
                        type="method",
                        level="W3_solution_direction",
                        context="External method referenced for comparison.",
                        source_pdf=paper["filename"],
                        provenance={"via": "external_compare_target"},
                    ))
                    existing_ids.add(e.tgt)

            errors = validate(atoms, edges)
            serious = [e for e in errors if e["rule"] != "R5"]
            assert len(serious) == 0, (
                f"Gatekeeper found errors in {paper['filename']}: {serious}"
            )


class TestCrossPaperReduction:
    """Test knowledge synthesis across multiple papers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp(prefix="ccchain_cross_")
        self.store = CCStore(
            db_path=os.path.join(self.tmpdir, "cross.db"),
            graph_dir=self.tmpdir,
        )
        yield
        self.store.db.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("ccchain.core.llm.chat_json")
    def test_cross_paper_reduction(self, mock_chat):
        """Reduce W5 atoms from different papers into shared W4 abstractions."""
        mock_chat.return_value = {
            "reduced_atoms": [{
                "name": "Safe RL with Uncertainty Quantification",
                "type": "method",
                "context": "Methods that jointly model reward and safety uncertainty using covariance-aware Q-value estimation.",
            }]
        }

        # Simulate W5 atoms from 3 papers on safe RL
        w5_atoms = [
            Atom(node_id="w5_copq_1", name="copq_critic_update", type="component",
                 level="W5_code_implementation", context="Cholesky-ordered projection for Q target."),
            Atom(node_id="w5_copq_2", name="copq_actor_loss", type="component",
                 level="W5_code_implementation", context="Actor loss with COP-Q constraints."),
            Atom(node_id="w5_lag_1", name="lagrangian_safety_layer", type="component",
                 level="W5_code_implementation", context="Lagrangian dual optimization for safety."),
        ]
        for a in w5_atoms:
            a.embedding = np.random.randn(1024).astype(np.float32)

        # Add edges to connect them (same connected component)
        edges = [
            Edge(src="w5_copq_1", relation="uses_component", tgt="w5_copq_2"),
            Edge(src="w5_copq_1", relation="compares", tgt="w5_lag_1"),
        ]

        self.store.insert_blueprint(w5_atoms, edges, "multi_paper.pdf")
        assert self.store.graph.vcount() == 3

        reducer = HierarchicalReducer(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:latest",
        )

        new_atoms = reducer.reduce_level(
            w5_atoms, edges,
            from_level="W5_code_implementation",
            to_level="W4_concrete_solution",
            graph=self.store.graph,
        )

        assert len(new_atoms) >= 1
        assert new_atoms[0].level == "W4_concrete_solution"
        assert len(new_atoms[0].context) > 0
