"""Flux-Insight Claim Chain v2 ontology: atom types, edge types, compatibility, validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 8 CC Atom Types
# ---------------------------------------------------------------------------
ATOM_TYPES: list[str] = [
    "method",       # algorithm / model / architecture
    "bottleneck",   # problem / challenge / limitation
    "paper",        # publication reference
    "fact",         # experimental fact / numeric result
    "component",    # module / sub-component
    "hypothesis",   # assumption / claim
    "experiment",   # experimental setup / evaluation protocol
    "verification", # proof / evidence / validation result
]

ATOM_TYPE_SET: set[str] = set(ATOM_TYPES)

# ---------------------------------------------------------------------------
# 11 CC Edge Types (categorised)
# ---------------------------------------------------------------------------
STRONG_CAUSAL: set[str] = {"EXTENDS", "IMPROVES", "REPLACES", "ADAPTS"}
WEAK_ASSOC: set[str] = {"USES_COMPONENT", "COMPARES"}
SEMANTIC: set[str] = {"BACKGROUND", "IMPLEMENTS", "VALIDATES", "BOUNDARY_OF", "RELATED_TO"}

EDGE_TYPES: list[str] = sorted(STRONG_CAUSAL | WEAK_ASSOC | SEMANTIC)
EDGE_TYPE_SET: set[str] = set(EDGE_TYPES)

# ---------------------------------------------------------------------------
# 14 Bottleneck Categories
# ---------------------------------------------------------------------------
BOTTLENECK_CATEGORIES: list[str] = [
    "overestimation_bias", "training_instability", "sample_inefficiency",
    "exploration_exploitation", "credit_assignment", "catastrophic_forgetting",
    "scalability", "communication_overhead", "non_stationarity",
    "partial_observability", "multi_objective_conflict",
    "representational_limitation", "computational_cost", "generalization_gap",
]

# ---------------------------------------------------------------------------
# Type-compatibility matrix: edge_type -> set of (src_type, tgt_type) tuples
# ---------------------------------------------------------------------------
EDGE_COMPAT: dict[str, set[tuple[str, str]]] = {
    "EXTENDS":        {("method", "method")},
    "IMPROVES":       {("method", "method"), ("method", "bottleneck")},
    "REPLACES":       {("method", "method"), ("component", "component")},
    "ADAPTS":         {("method", "method"), ("method", "bottleneck")},
    "USES_COMPONENT": {("method", "component")},
    "COMPARES":       {("method", "method"), ("experiment", "method")},
    "BACKGROUND":     {("paper", "method")},  # also wildcard below
    "IMPLEMENTS":     {("method", "hypothesis"), ("method", "paper")},
    "VALIDATES":      {("experiment", "hypothesis"), ("verification", "hypothesis"), ("method", "method")},
    "BOUNDARY_OF":    {("fact", "method"), ("fact", "experiment")},
    "RELATED_TO":     set(),  # wildcard: any -> any
}


# ---------------------------------------------------------------------------
# Rho evidence record (required for strong-causal edges)
# ---------------------------------------------------------------------------
@dataclass
class Rho:
    bottleneck: str       # must reference a bottleneck-type atom
    mechanism: str        # how the src addresses this bottleneck
    tradeoff: str         # cost / limitation introduced
    confidence: float     # 0.0 – 1.0

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
# Extraction output dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Atom:
    name: str
    type: str            # one of ATOM_TYPES
    context: str = ""    # original text snippet where this atom appears

    def __post_init__(self):
        self.type = self.type.lower()
        if self.type not in ATOM_TYPE_SET:
            raise ValueError(f"Invalid atom type: {self.type!r}")

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "context": self.context}

    @classmethod
    def from_dict(cls, d: dict) -> "Atom":
        return cls(
            name=str(d.get("name", "")),
            type=str(d.get("type", "")),
            context=str(d.get("context", "")),
        )


@dataclass
class Edge:
    src: str             # atom name
    relation: str        # one of EDGE_TYPES
    tgt: str             # atom name
    rho: Rho | None = None

    def __post_init__(self):
        self.relation = self.relation.upper()
        if self.relation not in EDGE_TYPE_SET:
            raise ValueError(f"Invalid edge type: {self.relation!r}")

    def to_dict(self) -> dict:
        d: dict = {"src": self.src, "relation": self.relation, "tgt": self.tgt}
        if self.rho is not None:
            d["rho"] = self.rho.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        rho = Rho.from_dict(d["rho"]) if "rho" in d and d["rho"] else None
        return cls(src=d["src"], relation=d["relation"], tgt=d["tgt"], rho=rho)


@dataclass
class Blueprint:
    atoms: list[Atom] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "atoms": [a.to_dict() for a in self.atoms],
            "edges": [e.to_dict() for e in self.edges],
        }


# ---------------------------------------------------------------------------
# OntologyGatekeeper — pre-write validation (R1 + R4)
# ---------------------------------------------------------------------------
class OntologyGatekeeper:
    """Validates atoms and edges against the Flux-Insight CC ontology."""

    def validate_entity(self, name: str) -> bool:
        if len(name) < 2:
            return False
        if name.isdigit():
            return False
        cleaned = re.sub(r"[^A-Za-z0-9 ]", " ", name).strip()
        return bool(cleaned)

    def validate_atom(self, atom: Atom) -> list[str]:
        errors: list[str] = []
        if not self.validate_entity(atom.name):
            errors.append(f"Atom name invalid: {atom.name!r}")
        if atom.type not in ATOM_TYPE_SET:
            errors.append(f"Atom type invalid: {atom.type!r}")
        return errors

    def validate_edge(
        self,
        edge: Edge,
        atom_map: dict[str, Atom] | None = None,
    ) -> list[str]:
        errors: list[str] = []

        # R1: reference integrity — endpoints must exist in atom map (if provided)
        if atom_map is not None:
            if edge.src not in atom_map:
                errors.append(f"Edge src {edge.src!r} not found in atoms")
            if edge.tgt not in atom_map:
                errors.append(f"Edge tgt {edge.tgt!r} not found in atoms")

        # R4: strong-causal edges must carry Rho evidence
        if edge.relation in STRONG_CAUSAL and edge.rho is None:
            errors.append(f"Strong-causal edge {edge.relation} requires Rho evidence")

        # Type compatibility
        if atom_map is not None:
            src_atom = atom_map.get(edge.src)
            tgt_atom = atom_map.get(edge.tgt)
            if src_atom and tgt_atom:
                pair = (src_atom.type, tgt_atom.type)
                compat = EDGE_COMPAT.get(edge.relation, set())
                # RELATED_TO is wildcard; BACKGROUND allows paper→method + any→any
                if edge.relation not in ("RELATED_TO", "BACKGROUND") and compat and pair not in compat:
                    errors.append(
                        f"Edge {edge.relation} incompatible: {pair[0]}→{pair[1]}"
                    )

        return errors

    def validate_blueprint(self, bp: Blueprint) -> list[str]:
        errors: list[str] = []
        atom_map: dict[str, Atom] = {a.name: a for a in bp.atoms}

        for atom in bp.atoms:
            errors.extend(self.validate_atom(atom))

        for edge in bp.edges:
            errors.extend(self.validate_edge(edge, atom_map))

        return errors
