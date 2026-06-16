"""CC atom ontology: 12 unified types, 4 levels (W2-W5), edges, compatibility, and dataclasses.

v0.3 design: type and level are 1:1 bound via TYPE_TO_LEVEL. Each type also
declares which CoE integrity checks (I1-I4) apply via TYPE_TO_COE_CHECKS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 12 CC Atom Types — organized by level (3 per level)
# ---------------------------------------------------------------------------
ATOM_TYPES: list[str] = [
    # W2 problem analysis
    "problem", "bottleneck", "hypothesis",
    # W3 solution direction
    "method", "citation", "concept",
    # W4 concrete solution
    "solution", "numerical", "conclusion",
    # W5 code implementation
    "component", "experiment", "verification",
]

ATOM_TYPE_SET: set[str] = set(ATOM_TYPES)

# ---------------------------------------------------------------------------
# Four knowledge levels (W2-W5)
# ---------------------------------------------------------------------------
LEVELS: list[str] = [
    "W2_problem_analysis",
    "W3_solution_direction",
    "W4_concrete_solution",
    "W5_code_implementation",
]

LEVEL_ORDER: dict[str, int] = {
    "W2_problem_analysis": 0,
    "W3_solution_direction": 1,
    "W4_concrete_solution": 2,
    "W5_code_implementation": 3,
}

LEVEL_ALIAS: dict[str, str] = {
    "W2": "W2_problem_analysis",
    "W3": "W3_solution_direction",
    "W4": "W4_concrete_solution",
    "W5": "W5_code_implementation",
}

# ---------------------------------------------------------------------------
# Type ↔ Level binding (1:1)
# ---------------------------------------------------------------------------
TYPE_TO_LEVEL: dict[str, str] = {
    "problem":      "W2_problem_analysis",
    "bottleneck":   "W2_problem_analysis",
    "hypothesis":   "W2_problem_analysis",
    "method":       "W3_solution_direction",
    "citation":     "W3_solution_direction",
    "concept":      "W3_solution_direction",
    "solution":     "W4_concrete_solution",
    "numerical":    "W4_concrete_solution",
    "conclusion":   "W4_concrete_solution",
    "component":    "W5_code_implementation",
    "experiment":   "W5_code_implementation",
    "verification": "W5_code_implementation",
}

# ---------------------------------------------------------------------------
# Type → CoE integrity checks triggered
# ---------------------------------------------------------------------------
TYPE_TO_COE_CHECKS: dict[str, set[str]] = {
    "numerical":    {"I1"},
    "citation":     {"I3"},
    "method":       {"I4"},
    "solution":     {"I4"},
    "experiment":   {"I2"},
    # problem/bottleneck/hypothesis/concept/conclusion/component/verification → no CoE checks
}

# ---------------------------------------------------------------------------
# v0.2 → v0.3 type migration (idempotent)
# ---------------------------------------------------------------------------
TYPE_MIGRATION_V02_TO_V03: dict[str, str] = {
    "paper": "citation",
    "fact": "concept",
    # method/solution: context-dependent, handled separately in store migration
}

# ---------------------------------------------------------------------------
# 11 CC Edge Types (horizontal, same-level)
# ---------------------------------------------------------------------------
STRONG_CAUSAL_EDGES: set[str] = {"extends", "improves", "replaces", "adapts"}

CC_EDGE_TYPES: list[str] = [
    "extends",
    "improves",
    "replaces",
    "adapts",
    "uses_component",
    "compares",
    "background",
    "implements",
    "validates",
    "boundary_of",
    "related_to",
]

# Hierarchical edges (cross-level)
HIERARCHY_EDGES: list[str] = ["aggregates_to", "decomposes_into"]

# ---------------------------------------------------------------------------
# 14 Bottleneck Categories (used in Rho.evidence for bottleneck-typed atoms)
# ---------------------------------------------------------------------------
BOTTLENECK_CATEGORIES: list[str] = [
    "overestimation_bias",
    "training_instability",
    "sample_inefficiency",
    "exploration_exploitation",
    "credit_assignment",
    "catastrophic_forgetting",
    "scalability",
    "communication_overhead",
    "non_stationarity",
    "partial_observability",
    "multi_objective_conflict",
    "representational_limitation",
    "computational_cost",
    "generalization_gap",
]

# ---------------------------------------------------------------------------
# Type-compatibility matrix: edge_type -> set of (src_type, tgt_type) tuples
# ---------------------------------------------------------------------------
TYPE_COMPATIBILITY: dict[str, set[tuple[str, str]]] = {
    "extends": {
        ("method", "method"),
        ("solution", "solution"),
        ("component", "component"),
    },
    "improves": {
        ("method", "method"),
        ("method", "bottleneck"),
        ("solution", "solution"),
        ("solution", "bottleneck"),
    },
    "replaces": {
        ("method", "method"),
        ("solution", "solution"),
        ("component", "component"),
        ("citation", "citation"),
    },
    "adapts": {
        ("method", "method"),
        ("solution", "solution"),
        ("method", "bottleneck"),
        ("solution", "bottleneck"),
    },
    "uses_component": {
        ("method", "component"),
        ("solution", "component"),
    },
    "compares": {
        ("method", "method"),
        ("solution", "solution"),
        ("experiment", "method"),
        ("experiment", "solution"),
    },
    "background": {
        ("citation", "method"),
        ("citation", "solution"),
        ("citation", "concept"),
    },
    "implements": {
        ("solution", "method"),
        ("component", "solution"),
        ("component", "method"),
        ("solution", "hypothesis"),
    },
    "validates": {
        ("verification", "hypothesis"),
        ("experiment", "hypothesis"),
        ("verification", "numerical"),
        ("verification", "method"),
        ("verification", "solution"),
    },
    "boundary_of": {
        ("concept", "method"),
        ("concept", "solution"),
        ("concept", "experiment"),
        ("numerical", "experiment"),
    },
    "related_to": set(),  # wildcard: any → any
}


# ---------------------------------------------------------------------------
# Rho evidence record (required for strong-causal edges)
# ---------------------------------------------------------------------------
@dataclass
class Rho:
    bottleneck: str
    mechanism: str
    tradeoff: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "bottleneck": self.bottleneck,
            "mechanism": self.mechanism,
            "tradeoff": self.tradeoff,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rho":
        return cls(
            bottleneck=str(d.get("bottleneck", "")),
            mechanism=str(d.get("mechanism", "")),
            tradeoff=str(d.get("tradeoff", "")),
            confidence=float(d.get("confidence", 0.7)),
        )


# ---------------------------------------------------------------------------
# TaskSpec — external acceptance criteria for I2 (Specification Violation)
# Mirrors ScientistOne §6 ADRS benchmark role: each task provides a fixed
# evaluator + starter code + scoring metric. In ccchain, the caller plays ADRS.
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    task_name: str
    eval_harness: str                # e.g. "smac-v1", "montezuma"
    success_criteria: str            # e.g. "win_rate > 0.9 against built-in AI"
    constraints: list[str] = field(default_factory=list)  # e.g. ["CTDE", "no centralised critic"]

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "eval_harness": self.eval_harness,
            "success_criteria": self.success_criteria,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(
            task_name=str(d.get("task_name", "")),
            eval_harness=str(d.get("eval_harness", "")),
            success_criteria=str(d.get("success_criteria", "")),
            constraints=list(d.get("constraints", [])),
        )


# ---------------------------------------------------------------------------
# Atom — the fundamental knowledge unit
# ---------------------------------------------------------------------------
@dataclass
class Atom:
    node_id: str
    name: str
    type: str  # ∈ ATOM_TYPES
    level: str  # ∈ LEVELS, must match TYPE_TO_LEVEL[type]
    context: str = ""
    version: int = 1
    source_pdf: str | None = None
    source_chunk: int | None = None
    code_ref: str | None = None
    references: dict | None = None
    tags: list[str] | None = None
    status: str = "active"  # lifecycle: active|needs_review|stuck|merged|transient
                           # audit: verified|demoted|low_reliability|low_confidence|skipped
    embedding: "np.ndarray | None" = None
    created_at: str | None = None
    updated_at: str | None = None
    code_body: str | None = None
    source_refs: list | None = None
    provenance: dict | None = None
    rowid: int | None = None  # SQLite implicit rowid; populated on load, not serialized

    def __post_init__(self):
        if self.type not in ATOM_TYPE_SET:
            raise ValueError(f"Invalid atom type: {self.type!r}")
        if self.level not in LEVEL_ORDER:
            raise ValueError(f"Invalid level: {self.level!r}")
        expected_level = TYPE_TO_LEVEL[self.type]
        if self.level != expected_level:
            raise ValueError(
                f"Type-level mismatch: type={self.type!r} requires level={expected_level!r}, "
                f"got {self.level!r}"
            )

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id,
            "name": self.name,
            "type": self.type,
            "level": self.level,
            "context": self.context,
            "version": self.version,
            "source_pdf": self.source_pdf,
            "source_chunk": self.source_chunk,
            "code_ref": self.code_ref,
            "references": self.references,
            "tags": self.tags,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "code_body": self.code_body,
            "source_refs": self.source_refs,
            "provenance": self.provenance,
        }
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Atom":
        return cls(
            node_id=str(d.get("node_id", "")),
            name=str(d.get("name", "")),
            type=str(d.get("type", "")),
            level=str(d.get("level", "")),
            context=str(d.get("context", "")),
            version=int(d.get("version", 1)),
            source_pdf=d.get("source_pdf"),
            source_chunk=d.get("source_chunk"),
            code_ref=d.get("code_ref"),
            references=d.get("references"),
            tags=d.get("tags"),
            status=str(d.get("status", "active")),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            code_body=d.get("code_body"),
            source_refs=d.get("source_refs"),
            provenance=d.get("provenance"),
            rowid=d.get("rowid"),
        )


# ---------------------------------------------------------------------------
# Edge — relationship between atoms
# ---------------------------------------------------------------------------
@dataclass
class Edge:
    src: str       # source atom node_id
    relation: str  # ∈ CC_EDGE_TYPES ∪ HIERARCHY_EDGES
    tgt: str       # target atom node_id
    weight: float = 1.0
    rho: Rho | None = None  # required for strong-causal edges
    provenance: dict | None = None
    rowid: int | None = None  # SQLite implicit rowid; populated on load, not serialized

    def __post_init__(self):
        all_edges = set(CC_EDGE_TYPES) | set(HIERARCHY_EDGES)
        if self.relation not in all_edges:
            raise ValueError(f"Invalid edge relation: {self.relation!r}")

    def to_dict(self) -> dict:
        d: dict = {
            "src": self.src,
            "relation": self.relation,
            "tgt": self.tgt,
            "weight": self.weight,
        }
        if self.rho is not None:
            d["rho"] = self.rho.to_dict()
        if self.provenance is not None:
            d["provenance"] = self.provenance
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        rho = Rho.from_dict(d["rho"]) if "rho" in d and d["rho"] else None
        return cls(
            src=str(d.get("src", "")),
            relation=str(d.get("relation", "")),
            tgt=str(d.get("tgt", "")),
            weight=float(d.get("weight", 1.0)),
            rho=rho,
            provenance=d.get("provenance"),
            rowid=d.get("rowid"),
        )


# ---------------------------------------------------------------------------
# Trajectory — temporary W2→W3→W4→W5 path, built on demand, destroyed after use
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    W2_problem: Atom | None = None
    W3_solutions: list[Atom] = field(default_factory=list)
    W4_implementations: list[Atom] = field(default_factory=list)
    W5_code: list[Atom] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    source_pdf: str = ""

    def get_embeddings_by_level(self) -> dict[str, "np.ndarray"]:
        """Return {level: (n, d) array} for atoms that have embeddings."""
        import numpy as np

        result: dict[str, list] = {"W2": [], "W3": [], "W4": [], "W5": []}
        if self.W2_problem is not None and self.W2_problem.embedding is not None:
            result["W2"].append(self.W2_problem.embedding)
        for a in self.W3_solutions:
            if a.embedding is not None:
                result["W3"].append(a.embedding)
        for a in self.W4_implementations:
            if a.embedding is not None:
                result["W4"].append(a.embedding)
        for a in self.W5_code:
            if a.embedding is not None:
                result["W5"].append(a.embedding)
        return {
            k: np.stack(v) if v else np.empty((0, 1024))
            for k, v in result.items()
        }

    def to_dict(self) -> dict:
        return {
            "W2_problem": self.W2_problem.to_dict() if self.W2_problem else None,
            "W3_solutions": [a.to_dict() for a in self.W3_solutions],
            "W4_implementations": [a.to_dict() for a in self.W4_implementations],
            "W5_code": [a.to_dict() for a in self.W5_code],
            "edges": [e.to_dict() for e in self.edges],
            "source_pdf": self.source_pdf,
        }
