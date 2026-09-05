"""
M2.4 — Persistence Layer
PS 26152 — AI-Powered Criminal Network Analysis System

Serializes a CriminalGraph to SQLite and reloads it with zero data loss.

Schema:
  nodes(entity_id TEXT PK, type, name, aliases JSON, source_documents JSON,
        confidence REAL, needs_review INT)
  edges(id INT PK, source TEXT, target TEXT, relationship TEXT,
        timestamp TEXT, source_record TEXT, confidence REAL, weight REAL)

Round-trip guarantee: same node/edge count and all attribute values preserved.
Corrupt/missing file raises a clear RuntimeError — no silent empty returns.
"""

import json
import logging
import sqlite3
from pathlib import Path

import networkx as nx

from M2.graph_builder import CriminalGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    entity_id        TEXT PRIMARY KEY,
    type             TEXT NOT NULL,
    name             TEXT NOT NULL,
    aliases          TEXT,      -- JSON array
    source_documents TEXT,      -- JSON array
    confidence       REAL NOT NULL,
    needs_review     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source           TEXT NOT NULL,
    target           TEXT NOT NULL,
    relationship     TEXT NOT NULL,
    timestamp        TEXT,
    source_record    TEXT,
    confidence       REAL NOT NULL,
    weight           REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (source) REFERENCES nodes(entity_id),
    FOREIGN KEY (target) REFERENCES nodes(entity_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_source_record ON edges(source_record);
"""


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def persist_graph(graph: CriminalGraph, db_path: str) -> None:
    """
    Serialize the graph to SQLite at `db_path`.

    Overwrites any existing tables (DROP + CREATE) to ensure a clean state.
    Wraps the write in a single transaction — either all rows are written or none.

    Args:
        graph   : the CriminalGraph to serialize
        db_path : path to the SQLite file (created if not exists)
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        # Drop and recreate for clean overwrite
        cur.executescript("DROP TABLE IF EXISTS edges; DROP TABLE IF EXISTS nodes;")
        cur.executescript(_DDL)

        # Insert nodes
        node_rows = []
        for node_id, attrs in graph.nodes(data=True):
            node_rows.append((
                node_id,
                attrs.get("type", ""),
                attrs.get("name", ""),
                json.dumps(attrs.get("aliases", []), ensure_ascii=False),
                json.dumps(attrs.get("source_documents", []), ensure_ascii=False),
                attrs.get("confidence", 0.0),
                int(attrs.get("needs_review", False)),
            ))
        cur.executemany(
            "INSERT OR REPLACE INTO nodes "
            "(entity_id, type, name, aliases, source_documents, confidence, needs_review) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            node_rows,
        )

        # Insert edges
        edge_rows = []
        for u, v, data in graph.edges(data=True):
            edge_rows.append((
                u, v,
                data.get("relationship", "ASSOCIATED_WITH"),
                data.get("timestamp", ""),
                data.get("source_record", ""),
                data.get("confidence", 0.0),
                data.get("weight", 1.0),
            ))
        cur.executemany(
            "INSERT INTO edges "
            "(source, target, relationship, timestamp, source_record, confidence, weight) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            edge_rows,
        )

        conn.commit()
        logger.info(
            f"[Persist] Saved {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges → {path}"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

def load_persisted_graph(db_path: str) -> CriminalGraph:
    """
    Reload a graph that was previously serialized by `persist_graph`.

    Raises:
        RuntimeError if the file doesn't exist, is empty, or has no nodes table.

    Returns:
        A fully reconstructed CriminalGraph.
    """
    path = Path(db_path)
    if not path.exists():
        raise RuntimeError(f"Graph database not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Graph database file is empty: {path}")

    graph: CriminalGraph = nx.MultiDiGraph()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        # Verify the schema exists
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "nodes" not in tables or "edges" not in tables:
            raise RuntimeError(
                f"Database {path} is missing required tables. "
                f"Found: {tables}. Was it created by persist_graph()?"
            )

        # Load nodes
        for row in conn.execute(
            "SELECT entity_id, type, name, aliases, source_documents, confidence, needs_review "
            "FROM nodes"
        ):
            graph.add_node(
                row["entity_id"],
                entity_id=row["entity_id"],
                type=row["type"],
                name=row["name"],
                aliases=json.loads(row["aliases"] or "[]"),
                source_documents=json.loads(row["source_documents"] or "[]"),
                confidence=row["confidence"],
                needs_review=bool(row["needs_review"]),
            )

        # Load edges
        for row in conn.execute(
            "SELECT source, target, relationship, timestamp, source_record, "
            "confidence, weight FROM edges"
        ):
            graph.add_edge(
                row["source"], row["target"],
                relationship=row["relationship"],
                timestamp=row["timestamp"],
                source_record=row["source_record"],
                confidence=row["confidence"],
                weight=row["weight"],
            )

        logger.info(
            f"[Persist] Loaded {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges ← {path}"
        )
    finally:
        conn.close()

    return graph
