"""Test CoEClaimVerifier — I1/I2/I3/I4 + CPR aggregation."""

from unittest.mock import patch

import pytest

from ccchain.core.ontology import Atom, Edge, TaskSpec
from ccchain.plugins.verification import CoEClaimVerifier


def _verifier():
    return CoEClaimVerifier(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:latest",
        reference_api_timeout=0.1,
        reference_api_max_retries=1,
    )


# ---------------------------------------------------------------------------
# I1 — Score Verification
# ---------------------------------------------------------------------------
@patch("ccchain.core.llm.chat_json")
def test_i1_score_within_tolerance(mock_chat):
    """numerical atom, LLM re-extract within tolerance → status='verified'."""
    mock_chat.return_value = {"score": 0.93}
    a = Atom(node_id="n1", name="Win rate", type="numerical",
             level="W4_concrete_solution",
             context="Our method achieves 0.95 win rate.",
             provenance={"score": 0.95, "score_std": 0.01})
    report = _verifier().verify([a], [])
    assert a.status == "verified"
    assert report["per_atom"][0]["checks"]["I1"]["status"] == "passed"


@patch("ccchain.core.llm.chat_json")
def test_i1_score_outside_tolerance(mock_chat):
    """numerical atom, LLM re-extract far from recorded → status='low_confidence'."""
    mock_chat.return_value = {"score": 0.55}
    a = Atom(node_id="n1", name="Win rate", type="numerical",
             level="W4_concrete_solution",
             context="Our method achieves 0.95 win rate.",
             provenance={"score": 0.95, "score_std": 0.0})
    report = _verifier().verify([a], [])
    assert a.status == "low_confidence"
    assert report["failures_by_check"]["I1"] == 1


@patch("ccchain.core.llm.chat_json")
def test_i1_cpr_calculation(mock_chat):
    """CPR = verified / (verified + low_confidence + low_reliability) for numerical atoms."""
    mock_chat.return_value = {"score": 0.95}  # all pass
    atoms = [
        Atom(node_id=f"n{i}", name=f"N{i}", type="numerical",
             level="W4_concrete_solution",
             context=f"Score 0.95", provenance={"score": 0.95})
        for i in range(3)
    ]
    report = _verifier().verify(atoms, [])
    assert report["cpr"] == 1.0
    assert report["atoms_audited"] == 3


def test_i1_skipped_without_score():
    """numerical atom without score → I1 skipped, no failure."""
    a = Atom(node_id="n1", name="N", type="numerical",
             level="W4_concrete_solution",
             provenance={"raw_citation": "x"})  # no score key
    # Manually mark as having score_key missing via dict shape
    a.provenance = {"note": "no score"}
    report = _verifier().verify([a], [])
    assert a.status == "verified"  # nothing failed, soft status
    # I1 status is skipped, not failed
    assert report["per_atom"][0]["checks"]["I1"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# I2 — Specification Violation
# ---------------------------------------------------------------------------
@patch("ccchain.core.llm.chat_json_majority")
def test_i2_violates_spec(mock_majority):
    """experiment atom violates task_spec → status='low_reliability'."""
    mock_majority.return_value = {"verdict": "violates_spec", "reasoning": "uses centralized critic"}
    a = Atom(node_id="e1", name="CTDE Exp", type="experiment",
             level="W5_code_implementation",
             context="Training loop",
             code_body="critic = CentralizedCritic()",
             provenance={"code_span": "lines 1-50"})
    ts = TaskSpec(task_name="marl-1v1", eval_harness="smac-v1",
                  success_criteria="win > 0.9", constraints=["CTDE"])
    report = _verifier().verify([a], [], task_spec=ts)
    assert a.status == "low_reliability"
    assert report["failures_by_check"]["I2"] == 1


@patch("ccchain.core.llm.chat_json_majority")
def test_i2_compliant(mock_majority):
    """experiment atom compliant → status='verified'."""
    mock_majority.return_value = {"verdict": "compliant", "reasoning": "follows CTDE"}
    a = Atom(node_id="e1", name="CTDE Exp", type="experiment",
             level="W5_code_implementation",
             code_body="critic = DecentralizedCritic()",
             provenance={"code_span": "1-50"})
    ts = TaskSpec(task_name="marl-1v1", eval_harness="smac-v1",
                  success_criteria="win > 0.9", constraints=["CTDE"])
    report = _verifier().verify([a], [], task_spec=ts)
    assert a.status == "verified"


def test_i2_skipped_without_task_spec():
    """No task_spec → I2 skipped, atom verified (if no other checks)."""
    a = Atom(node_id="e1", name="Exp", type="experiment",
             level="W5_code_implementation",
             provenance={"code_span": "1-50"})
    report = _verifier().verify([a], [])  # no task_spec
    assert a.status == "verified"  # nothing failed
    assert report["per_atom"][0]["checks"]["I2"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# I3 — Reference Verification
# ---------------------------------------------------------------------------
@patch("ccchain.core.references.requests.get")
def test_i3_dangling_citation(mock_get):
    """citation atom, all APIs miss → status='low_reliability'."""
    from unittest.mock import MagicMock
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"data": []}  # sem scholar miss
    m.text = "<feed></feed>"  # arxiv miss
    mock_get.return_value = m

    # Sequential mocks for each API
    from unittest.mock import MagicMock
    def side(url, **kwargs):
        m = MagicMock()
        m.status_code = 200
        if "semanticscholar" in url:
            m.json.return_value = {"data": []}
        elif "arxiv" in url:
            m.text = "<feed></feed>"
        elif "openalex" in url:
            m.json.return_value = {"results": []}
        elif "crossref" in url:
            m.json.return_value = {"message": {"items": []}}
        return m
    mock_get.side_effect = side

    a = Atom(node_id="c1", name="Fake citation", type="citation",
             level="W3_solution_direction",
             context="Smith et al. 2099 definitely fake paper",
             provenance={"raw_citation": "Smith et al., definitely fake, 2099"})
    report = _verifier().verify([a], [])
    assert a.status == "low_reliability"
    assert report["failures_by_check"]["I3"] == 1


@patch("ccchain.core.references.requests.get")
def test_i3_resolved_passes(mock_get):
    """citation atom, Semantic Scholar hit → status='verified'."""
    from unittest.mock import MagicMock
    def side(url, **kwargs):
        m = MagicMock()
        m.status_code = 200
        if "semanticscholar" in url:
            m.json.return_value = {"data": [{
                "title": "Real Paper",
                "year": 2020,
                "externalIds": {"DOI": "10.1/real"},
            }]}
        return m
    mock_get.side_effect = side

    a = Atom(node_id="c1", name="Real citation", type="citation",
             level="W3_solution_direction",
             context="Vaswani et al. 2017",
             provenance={"raw_citation": "Vaswani et al. 2017 Attention"})
    report = _verifier().verify([a], [])
    assert a.status == "verified"
    assert a.provenance["resolved"]["title"] == "Real Paper"


# ---------------------------------------------------------------------------
# I4 — Method-Code Alignment
# ---------------------------------------------------------------------------
@patch("ccchain.core.llm.chat_json_majority")
def test_i4_misaligned_code(mock_majority):
    """method atom with misaligned child component → status='low_reliability'."""
    mock_majority.return_value = {"verdict": "misaligned", "reasoning": "uses Shapley not OT"}
    method = Atom(node_id="m1", name="OT method", type="method",
                  level="W3_solution_direction",
                  context="Optimal transport credit assignment",
                  provenance={"code_span": "method section"})
    child = Atom(node_id="c1", name="Shapley impl", type="component",
                 level="W5_code_implementation",
                 code_body="def shapley(): ...",
                 context="Shapley value sampler")
    edge = Edge(src="m1", relation="decomposes_into", tgt="c1")
    report = _verifier().verify([method, child], [edge])
    method_report = next(p for p in report["per_atom"] if p["node_id"] == "m1")
    assert method_report["status"] == "low_reliability"
    assert report["failures_by_check"]["I4"] == 1


@patch("ccchain.core.llm.chat_json_majority")
def test_i4_aligned_passes(mock_majority):
    """method atom with aligned child component → status='verified'."""
    mock_majority.return_value = {"verdict": "aligned", "reasoning": "implements OT"}
    method = Atom(node_id="m1", name="OT method", type="method",
                  level="W3_solution_direction",
                  context="Optimal transport credit assignment",
                  provenance={"code_span": "method section"})
    child = Atom(node_id="c1", name="Sinkhorn impl", type="component",
                 level="W5_code_implementation",
                 code_body="def sinkhorn(): ...")
    edge = Edge(src="m1", relation="decomposes_into", tgt="c1")
    report = _verifier().verify([method, child], [edge])
    method_report = next(p for p in report["per_atom"] if p["node_id"] == "m1")
    assert method_report["status"] == "verified"


def test_i4_skipped_without_child_components():
    """method atom with no child components → I4 skipped."""
    method = Atom(node_id="m1", name="Method", type="method",
                  level="W3_solution_direction",
                  context="A method",
                  provenance={"code_span": "section 3"})
    report = _verifier().verify([method], [])
    method_report = next(p for p in report["per_atom"] if p["node_id"] == "m1")
    # I4 skipped, no other checks → status verified
    assert method_report["status"] == "verified"
    assert method_report["checks"]["I4"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_subjective_types_marked_skipped():
    """problem/bottleneck/hypothesis/concept/conclusion/component without experiment type → skipped."""
    atoms = [
        Atom(node_id="p1", name="P", type="problem", level="W2_problem_analysis"),
        Atom(node_id="b1", name="B", type="bottleneck", level="W2_problem_analysis"),
        Atom(node_id="c1", name="C", type="concept", level="W3_solution_direction"),
        Atom(node_id="cn1", name="CN", type="conclusion", level="W4_concrete_solution"),
        Atom(node_id="comp1", name="Comp", type="component", level="W5_code_implementation"),
    ]
    report = _verifier().verify(atoms, [])
    assert report["atoms_skipped"] == 5
    assert report["atoms_audited"] == 0
    assert all(a.status == "skipped" for a in atoms)


def test_demoted_atoms_skip_audit():
    """Atoms already marked status='demoted' are skipped by audit."""
    a = Atom(node_id="n1", name="Demoted", type="conclusion",
             level="W4_concrete_solution")
    a.status = "demoted"
    report = _verifier().verify([a], [])
    assert report["atoms_skipped"] == 1
    assert a.status == "demoted"  # unchanged


def test_report_summary_counts():
    """Report counts add up."""
    from unittest.mock import patch as _p

    @_p("ccchain.core.llm.chat_json")
    def inner(mock_chat):
        mock_chat.return_value = {"score": 0.5}
        atoms = [
            Atom(node_id="n1", name="Pass", type="numerical",
                 level="W4_concrete_solution",
                 context="score 0.5",
                 provenance={"score": 0.5}),
            Atom(node_id="p1", name="P", type="problem",
                 level="W2_problem_analysis"),
        ]
        report = _verifier().verify(atoms, [])
        assert report["atoms_audited"] + report["atoms_skipped"] == len(atoms)
        assert report["atoms_passed"] + report["atoms_failed"] == report["atoms_audited"]
        assert 0.0 <= report["cpr"] <= 1.0

    inner()
