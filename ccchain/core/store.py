"""Dual storage engine: SQLite (authoritative) + igraph (graph topology)."""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import time
from pathlib import Path

import igraph as ig
import numpy as np

from ccchain.core.ontology import (
    LEVEL_ORDER,
    TYPE_MIGRATION_V02_TO_V03,
    TYPE_TO_LEVEL,
    Atom,
    Edge,
    Trajectory,
)


class CCStore:
    """Dual-storage: SQLite for authoritative persistence, igraph for graph ops.

    Write strategy: SQLite first (transaction-protected) → igraph best-effort.
    On startup, _verify_consistency() auto-detects drift and rebuilds igraph.
    """

    def __init__(self, db_path: str, graph_dir: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)

        self.db_path = db_path
        self.graph_dir = graph_dir
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

        self.graph: ig.Graph = self._load_or_build_graph()
        self._node_index: dict[str, int] = {}
        self._embedding_cache: np.ndarray | None = None
        self._rebuild_node_index()
        self._verify_consistency()
        self._load_embeddings()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cc_nodes (
                node_id     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                type        TEXT NOT NULL,
                level       TEXT NOT NULL,
                context     TEXT DEFAULT '',
                version     INTEGER DEFAULT 1,
                source_pdf  TEXT,
                source_chunk INTEGER,
                code_ref    TEXT,
                refs        TEXT,
                tags        TEXT,
                status      TEXT DEFAULT 'active',
                embedding   BLOB,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                code_body   TEXT,
                source_refs TEXT,
                provenance  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_level ON cc_nodes(level);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON cc_nodes(type);
            CREATE INDEX IF NOT EXISTS idx_nodes_status ON cc_nodes(status);

            CREATE TABLE IF NOT EXISTS cc_edges (
                src         TEXT NOT NULL,
                tgt         TEXT NOT NULL,
                relation    TEXT NOT NULL,
                weight      REAL DEFAULT 1.0,
                rho_json    TEXT,
                provenance  TEXT,
                PRIMARY KEY (src, tgt, relation)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON cc_edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_tgt ON cc_edges(tgt);
        """)
        # Idempotent ALTER TABLE migration for existing databases pre-v0.2
        for col, decl in [
            ("code_body", "TEXT"),
            ("source_refs", "TEXT"),
            ("provenance", "TEXT"),
        ]:
            try:
                self.db.execute(f"ALTER TABLE cc_nodes ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # column already exists
        try:
            self.db.execute("ALTER TABLE cc_edges ADD COLUMN provenance TEXT")
        except sqlite3.OperationalError:
            pass
        self.db.commit()

        # v0.2 → v0.3 idempotent data migration (paper→citation, fact→concept, W4 method→solution)
        self._migrate_v02_to_v03_types()

    # ------------------------------------------------------------------
    # v0.2 → v0.3 type migration
    # ------------------------------------------------------------------
    def _migrate_v02_to_v03_types(self) -> int:
        """Idempotent migration of legacy atom types to v0.3 12-type system.

        Rules:
          - paper → citation (W3_solution_direction)
          - fact → concept (W3_solution_direction)
          - (method, W4_concrete_solution) → solution (level stays W4)
          - any atom whose (type, level) violates TYPE_TO_LEVEL → force canonical level

        Returns: number of rows migrated. Safe to call repeatedly.
        """
        migrated = 0
        # Pass 1: type renames (paper→citation, fact→concept)
        for old_t, new_t in TYPE_MIGRATION_V02_TO_V03.items():
            cur = self.db.execute(
                "UPDATE cc_nodes SET type = ?, level = ? WHERE type = ?",
                (new_t, TYPE_TO_LEVEL[new_t], old_t),
            )
            migrated += cur.rowcount

        # Pass 2: W4 method → solution
        cur = self.db.execute(
            "UPDATE cc_nodes SET type = 'solution' WHERE type = 'method' AND level = 'W4_concrete_solution'"
        )
        migrated += cur.rowcount

        # Pass 3: force type-level consistency per TYPE_TO_LEVEL
        for t, lvl in TYPE_TO_LEVEL.items():
            cur = self.db.execute(
                "UPDATE cc_nodes SET level = ? WHERE type = ? AND level != ?",
                (lvl, t, lvl),
            )
            migrated += cur.rowcount

        if migrated > 0:
            self.db.commit()
        return migrated

    # ------------------------------------------------------------------
    # Graph persistence
    # ------------------------------------------------------------------
    def _graph_pickle_path(self) -> str:
        return os.path.join(self.graph_dir, "cc_graph.pickle")

    def _load_or_build_graph(self) -> ig.Graph:
        pickle_path = self._graph_pickle_path()
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return self._build_graph_from_db()

    def _build_graph_from_db(self) -> ig.Graph:
        """Full rebuild of igraph from SQLite."""
        g = ig.Graph(directed=True)
        rows = self.db.execute("SELECT node_id, name, type, level FROM cc_nodes").fetchall()
        for row in rows:
            g.add_vertex(
                name=row["node_id"],
                label=row["name"],
                type=row["type"],
                level=row["level"],
            )
        edge_rows = self.db.execute(
            "SELECT src, tgt, relation, weight FROM cc_edges"
        ).fetchall()
        for row in edge_rows:
            try:
                src_idx = g.vs.find(name=row["src"]).index
                tgt_idx = g.vs.find(name=row["tgt"]).index
                g.add_edge(src_idx, tgt_idx, relation=row["relation"], weight=row["weight"])
            except (ValueError, KeyError):
                pass
        return g

    def persist(self):
        """Persist igraph to pickle and GraphML."""
        pickle_path = self._graph_pickle_path()
        with open(pickle_path, "wb") as f:
            pickle.dump(self.graph, f)
        graphml_path = os.path.join(self.graph_dir, "cc_graph.graphml")
        self.graph.write_graphml(graphml_path)

    def _rebuild_node_index(self):
        self._node_index = {}
        for v in self.graph.vs:
            self._node_index[v["name"]] = v.index

    def _verify_consistency(self):
        """Check SQLite vs igraph consistency; rebuild igraph if drifted."""
        sqlite_count = self.db.execute("SELECT COUNT(*) as c FROM cc_nodes").fetchone()["c"]
        igraph_count = self.graph.vcount()
        if sqlite_count != igraph_count:
            self.graph = self._build_graph_from_db()
            self._rebuild_node_index()

    def _load_embeddings(self):
        rows = self.db.execute(
            "SELECT node_id, embedding FROM cc_nodes WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            self._embedding_cache = np.empty((0, 1024), dtype=np.float32)
            return
        embeds = []
        for row in rows:
            embeds.append(np.frombuffer(row["embedding"], dtype=np.float32))
        self._embedding_cache = np.stack(embeds)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def query_by_level(
        self, level: str, type: str | None = None, status: str | None = "active"
    ) -> list[Atom]:
        query = "SELECT rowid, * FROM cc_nodes WHERE level = ?"
        params: list = [level]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        rows = self.db.execute(query, params).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def query_by_id(self, node_id: str) -> Atom | None:
        row = self.db.execute(
            "SELECT rowid, * FROM cc_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return self._row_to_atom(row) if row else None

    def query_by_domain(self, domain: str) -> list[Atom]:
        rows = self.db.execute(
            "SELECT rowid, * FROM cc_nodes WHERE tags LIKE ?", (f'%"domain:{domain}"%',)
        ).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def get_neighbors(
        self, node_id: str, relation: str | None = None, direction: str = "both"
    ) -> list[str]:
        neighbors: list[str] = []
        if direction in ("both", "out"):
            query = "SELECT tgt FROM cc_edges WHERE src = ?"
            params: list = [node_id]
            if relation:
                query += " AND relation = ?"
                params.append(relation)
            for row in self.db.execute(query, params):
                neighbors.append(row["tgt"])
        if direction in ("both", "in"):
            query = "SELECT src FROM cc_edges WHERE tgt = ?"
            params = [node_id]
            if relation:
                query += " AND relation = ?"
                params.append(relation)
            for row in self.db.execute(query, params):
                neighbors.append(row["src"])
        return list(set(neighbors))

    def get_all_embeddings(self) -> np.ndarray:
        if self._embedding_cache is not None:
            return self._embedding_cache
        self._load_embeddings()
        assert self._embedding_cache is not None
        return self._embedding_cache

    def query_embeddings_by_level(self, level: str) -> dict[str, "np.ndarray"]:
        """Return {node_id: vector} for all embedded atoms at `level`.

        Keyed by node_id (unlike get_all_embeddings, which is a positionally-
        aligned bare ndarray). Used by consolidate for safe pairwise cosine.
        """
        rows = self.db.execute(
            "SELECT node_id, embedding FROM cc_nodes "
            "WHERE level = ? AND embedding IS NOT NULL",
            (level,),
        ).fetchall()
        out: dict[str, np.ndarray] = {}
        for r in rows:
            out[r["node_id"]] = np.frombuffer(r["embedding"], dtype=np.float32)
        return out

    def get_all_trajectories(self, domain: str | None = None) -> list[Trajectory]:
        """Build all trajectories: start from W5 atoms, trace upward via PPR on AGGREGATES_TO edges."""
        from ccchain.core.graph import local_ppr_path

        if domain:
            w5_atoms = [
                a for a in self.query_by_level("W5_code_implementation")
                if not domain or any(
                    t == f"domain:{domain}" for t in (a.tags or [])
                )
            ]
        else:
            w5_atoms = self.query_by_level("W5_code_implementation")

        trajectories: list[Trajectory] = []
        seen_w5: set[str] = set()

        for w5 in w5_atoms:
            if w5.node_id in seen_w5:
                continue
            seen_w5.add(w5.node_id)

            if w5.node_id not in self._node_index:
                continue
            w5_idx = self._node_index[w5.node_id]

            # Trace upward level by level
            w4_idx = self._best_ancestor(w5_idx, "W4_concrete_solution")
            w3_idx = self._best_ancestor(w4_idx, "W3_solution_direction") if w4_idx >= 0 else -1
            w2_idx = self._best_ancestor(w3_idx, "W2_problem_analysis") if w3_idx >= 0 else -1

            traj = Trajectory(source_pdf=w5.source_pdf or "")
            traj.W5_code.append(w5)

            if w4_idx >= 0:
                w4 = self.query_by_id(self.graph.vs[w4_idx]["name"])
                if w4:
                    traj.W4_implementations.append(w4)
            if w3_idx >= 0:
                w3 = self.query_by_id(self.graph.vs[w3_idx]["name"])
                if w3:
                    traj.W3_solutions.append(w3)
            if w2_idx >= 0:
                w2 = self.query_by_id(self.graph.vs[w2_idx]["name"])
                if w2:
                    traj.W2_problem = w2

            trajectories.append(traj)

        return trajectories

    def _best_ancestor(self, child_idx: int, target_level: str) -> int:
        """Find the best ancestor at target_level via local PPR from child."""
        if child_idx < 0:
            return -1
        candidates = [
            v.index for v in self.graph.vs
            if v["level"] == target_level
        ]
        if not candidates:
            return -1
        from ccchain.core.graph import local_ppr_path

        return local_ppr_path(self.graph, child_idx, candidates)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def insert_blueprint(
        self,
        atoms: list[Atom],
        edges: list[Edge],
        source_pdf: str,
        embed_fn: "callable | None" = None,
    ) -> dict:
        """Insert a full blueprint: atoms + edges, with optional embedding.

        Returns {inserted_nodes, inserted_edges, duplicates_merged}.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        inserted_nodes = 0
        duplicates_merged = 0

        for atom in atoms:
            atom.source_pdf = atom.source_pdf or source_pdf
            atom.created_at = atom.created_at or now
            atom.updated_at = now

            existing = self.db.execute(
                "SELECT node_id FROM cc_nodes WHERE node_id = ?", (atom.node_id,)
            ).fetchone()
            if existing:
                duplicates_merged += 1
                continue

            embedding_blob = None
            if atom.embedding is not None:
                embedding_blob = np.asarray(atom.embedding, dtype=np.float32).tobytes()

            self.db.execute(
                """INSERT INTO cc_nodes
                   (node_id, name, type, level, context, version, source_pdf,
                    source_chunk, code_ref, refs, tags, status,
                    embedding, created_at, updated_at,
                    code_body, source_refs, provenance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    atom.node_id, atom.name, atom.type, atom.level,
                    atom.context, atom.version, atom.source_pdf,
                    atom.source_chunk, atom.code_ref,
                    json.dumps(atom.references) if atom.references else None,
                    json.dumps(atom.tags) if atom.tags else None,
                    atom.status,
                    embedding_blob,
                    atom.created_at, atom.updated_at,
                    atom.code_body,
                    json.dumps(atom.source_refs) if atom.source_refs else None,
                    json.dumps(atom.provenance) if atom.provenance else None,
                ),
            )
            inserted_nodes += 1

            # Add to igraph
            self.graph.add_vertex(
                name=atom.node_id,
                label=atom.name,
                type=atom.type,
                level=atom.level,
            )

        inserted_edges = 0
        for edge in edges:
            try:
                cursor = self.db.execute(
                    """INSERT OR IGNORE INTO cc_edges (src, tgt, relation, weight, rho_json, provenance)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        edge.src, edge.tgt, edge.relation, edge.weight,
                        json.dumps(edge.rho.to_dict()) if edge.rho else None,
                        json.dumps(edge.provenance) if edge.provenance else None,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted_edges += 1
                    try:
                        src_idx = self.graph.vs.find(name=edge.src).index
                        tgt_idx = self.graph.vs.find(name=edge.tgt).index
                        self.graph.add_edge(
                            src_idx, tgt_idx,
                            relation=edge.relation,
                            weight=edge.weight,
                        )
                    except (ValueError, KeyError):
                        pass
            except sqlite3.IntegrityError:
                pass

        self.db.commit()
        self._rebuild_node_index()
        self._load_embeddings()  # invalidate cache after insert
        self.persist()

        return {
            "inserted_nodes": inserted_nodes,
            "inserted_edges": inserted_edges,
            "duplicates_merged": duplicates_merged,
        }

    def upsert_atoms(self, atoms: list[Atom]) -> int:
        """Insert or update atoms. Returns number affected."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        count = 0
        for atom in atoms:
            atom.updated_at = now
            embedding_blob = None
            if atom.embedding is not None:
                embedding_blob = np.asarray(atom.embedding, dtype=np.float32).tobytes()

            self.db.execute(
                """INSERT INTO cc_nodes
                   (node_id, name, type, level, context, version, source_pdf,
                    source_chunk, code_ref, refs, tags, status,
                    embedding, created_at, updated_at,
                    code_body, source_refs, provenance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(node_id) DO UPDATE SET
                    name=excluded.name, type=excluded.type, level=excluded.level,
                    context=excluded.context, version=excluded.version,
                    tags=excluded.tags, status=excluded.status,
                    embedding=COALESCE(excluded.embedding, cc_nodes.embedding),
                    code_body=COALESCE(excluded.code_body, cc_nodes.code_body),
                    source_refs=COALESCE(excluded.source_refs, cc_nodes.source_refs),
                    provenance=COALESCE(excluded.provenance, cc_nodes.provenance),
                    updated_at=excluded.updated_at""",
                (
                    atom.node_id, atom.name, atom.type, atom.level,
                    atom.context, atom.version, atom.source_pdf,
                    atom.source_chunk, atom.code_ref,
                    json.dumps(atom.references) if atom.references else None,
                    json.dumps(atom.tags) if atom.tags else None,
                    atom.status,
                    embedding_blob,
                    atom.created_at or now, atom.updated_at,
                    atom.code_body,
                    json.dumps(atom.source_refs) if atom.source_refs else None,
                    json.dumps(atom.provenance) if atom.provenance else None,
                ),
            )
            count += 1

            try:
                vidx = self.graph.vs.find(name=atom.node_id).index
                self.graph.vs[vidx].update_attributes(
                    label=atom.name, type=atom.type, level=atom.level,
                )
            except (ValueError, KeyError):
                self.graph.add_vertex(
                    name=atom.node_id,
                    label=atom.name,
                    type=atom.type,
                    level=atom.level,
                )

        self.db.commit()
        self._rebuild_node_index()
        self._load_embeddings()
        return count

    def upsert_edges(self, edges: list[Edge]) -> int:
        """Insert or ignore edges. Returns number inserted."""
        count = 0
        for edge in edges:
            try:
                cursor = self.db.execute(
                    """INSERT OR IGNORE INTO cc_edges (src, tgt, relation, weight, rho_json, provenance)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        edge.src, edge.tgt, edge.relation, edge.weight,
                        json.dumps(edge.rho.to_dict()) if edge.rho else None,
                        json.dumps(edge.provenance) if edge.provenance else None,
                    ),
                )
                if cursor.rowcount > 0:
                    count += 1
                    try:
                        src_idx = self.graph.vs.find(name=edge.src).index
                        tgt_idx = self.graph.vs.find(name=edge.tgt).index
                        self.graph.add_edge(
                            src_idx, tgt_idx,
                            relation=edge.relation,
                            weight=edge.weight,
                        )
                    except (ValueError, KeyError):
                        pass
            except sqlite3.IntegrityError:
                pass
        self.db.commit()
        return count

    def edge_targets(self, src: str) -> list[tuple[str, str]]:
        """Return [(tgt, relation), ...] for outgoing edges of `src`."""
        rows = self.db.execute(
            "SELECT tgt, relation FROM cc_edges WHERE src = ?", (src,)
        ).fetchall()
        return [(r["tgt"], r["relation"]) for r in rows]

    def edge_sources(self, tgt: str) -> list[tuple[str, str]]:
        """Return [(src, relation), ...] for incoming edges of `tgt`."""
        rows = self.db.execute(
            "SELECT src, relation FROM cc_edges WHERE tgt = ?", (tgt,)
        ).fetchall()
        return [(r["src"], r["relation"]) for r in rows]

    def delete_edges_touching(self, node_ids: list[str]) -> int:
        """Delete every edge whose src or tgt is in `node_ids`. Returns rows deleted."""
        if not node_ids:
            return 0
        ph = ",".join("?" for _ in node_ids)
        cur = self.db.execute(
            f"DELETE FROM cc_edges WHERE src IN ({ph}) OR tgt IN ({ph})",
            node_ids + node_ids,
        )
        self.db.commit()
        return cur.rowcount

    def delete_transient(self) -> int:
        """Delete all atoms with status='transient' and their connected edges."""
        transient = self.db.execute(
            "SELECT node_id FROM cc_nodes WHERE status = 'transient'"
        ).fetchall()
        ids = [r["node_id"] for r in transient]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        self.db.execute(
            f"DELETE FROM cc_edges WHERE src IN ({placeholders}) OR tgt IN ({placeholders})",
            ids + ids,
        )
        self.db.execute(
            f"DELETE FROM cc_nodes WHERE node_id IN ({placeholders})", ids
        )
        self.db.commit()
        self.graph = self._build_graph_from_db()
        self._rebuild_node_index()
        return len(ids)

    def rebuild_graph(self):
        """Force full rebuild of igraph from SQLite."""
        self.graph = self._build_graph_from_db()
        self._rebuild_node_index()
        self.persist()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _row_to_atom(self, row: sqlite3.Row) -> Atom:
        return Atom(
            node_id=row["node_id"],
            name=row["name"],
            type=row["type"],
            level=row["level"],
            context=row["context"] or "",
            version=row["version"],
            source_pdf=row["source_pdf"],
            source_chunk=row["source_chunk"],
            code_ref=row["code_ref"],
            references=json.loads(row["refs"]) if row["refs"] else None,
            tags=json.loads(row["tags"]) if row["tags"] else None,
            status=row["status"] or "active",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            code_body=row["code_body"],
            source_refs=json.loads(row["source_refs"]) if row["source_refs"] else None,
            provenance=json.loads(row["provenance"]) if row["provenance"] else None,
            rowid=row["rowid"],
        )
