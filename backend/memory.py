"""
Agent 3 -- Memory Agent (Immune Memory).

Persists every AttackDNA ever seen, together with the responses that
were tried against it and how well each one worked. On a new DNA, does
a similarity search against everything in memory so that a
never-exactly-seen-before attack can still be matched to a known
threat *family* and reuse what has historically worked.

SQLite is used for the prototype (per the design doc's guidance to
avoid standing up Postgres/pgvector/Neo4j for a first version); the
schema mirrors what those would look like so it's a straightforward
lift later.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np

from dna import AttackDNA, NetworkGenome, BehavioralGenome, ResourceGenome, ContextGenome, EvolutionMeta, cosine_similarity

SCHEMA = """
CREATE TABLE IF NOT EXISTS attack_dna (
    dna_id TEXT PRIMARY KEY,
    attack_type TEXT,
    payload_json TEXT,
    vector_json TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS dna_lineage (
    child_id TEXT,
    parent_id TEXT,
    relation TEXT   -- 'mutation' | 'recombination'
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dna_id TEXT,
    action TEXT,
    success INTEGER,       -- 1/0
    security_benefit REAL,
    business_impact REAL,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    dna_id TEXT,
    anomaly_score REAL,
    matched_dna_id TEXT,
    similarity REAL,
    recommended_action TEXT,
    risk_level TEXT,
    status TEXT,           -- pending | approved | rejected | auto_executed
    source TEXT,           -- synthetic_telemetry | attack_lab
    scenario_id TEXT,
    created_at REAL
);
"""


class ImmuneMemory:
    def __init__(self, path: str = "cybergene.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- DNA storage ----------------------------------------------------
    def store_dna(self, dna: AttackDNA):
        vec = dna.vector()
        self.conn.execute(
            "INSERT OR REPLACE INTO attack_dna (dna_id, attack_type, payload_json, vector_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (dna.dna_id, dna.attack_type, json.dumps(dna.to_dict(), default=str), json.dumps(vec.tolist()), time.time()),
        )
        for parent in dna.evolution.parent_ids:
            relation = "recombination" if len(dna.evolution.parent_ids) > 1 else "mutation"
            self.conn.execute(
                "INSERT INTO dna_lineage (child_id, parent_id, relation) VALUES (?, ?, ?)",
                (dna.dna_id, parent, relation),
            )
        self.conn.commit()

    def get_dna_raw(self, dna_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT payload_json FROM attack_dna WHERE dna_id=?", (dna_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all_dna(self) -> List[dict]:
        rows = self.conn.execute("SELECT payload_json FROM attack_dna ORDER BY created_at DESC").fetchall()
        return [json.loads(r[0]) for r in rows]

    def lineage(self, dna_id: str) -> dict:
        parents = self.conn.execute(
            "SELECT parent_id, relation FROM dna_lineage WHERE child_id=?", (dna_id,)
        ).fetchall()
        children = self.conn.execute(
            "SELECT child_id, relation FROM dna_lineage WHERE parent_id=?", (dna_id,)
        ).fetchall()
        return {
            "dna_id": dna_id,
            "parents": [{"id": p, "relation": r} for p, r in parents],
            "children": [{"id": c, "relation": r} for c, r in children],
        }

    # ---- similarity search (the "immune memory recall") -----------------
    def similarity_search(self, dna: AttackDNA, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vec = dna.vector()
        rows = self.conn.execute("SELECT dna_id, vector_json FROM attack_dna").fetchall()
        scored = []
        for dna_id, vec_json in rows:
            v = np.array(json.loads(vec_json))
            sim = cosine_similarity(query_vec, v)
            scored.append((dna_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---- response outcome tracking --------------------------------------
    def record_response(self, dna_id: str, action: str, success: bool,
                         security_benefit: float, business_impact: float):
        self.conn.execute(
            "INSERT INTO responses (dna_id, action, success, security_benefit, business_impact, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (dna_id, action, int(success), security_benefit, business_impact, time.time()),
        )
        self.conn.commit()

    def response_stats(self, dna_id: str) -> dict:
        """Success rate per action historically tried against this exact DNA."""
        rows = self.conn.execute(
            "SELECT action, success FROM responses WHERE dna_id=?", (dna_id,)
        ).fetchall()
        stats: dict = {}
        for action, success in rows:
            s = stats.setdefault(action, {"tries": 0, "successes": 0})
            s["tries"] += 1
            s["successes"] += success
        for action, s in stats.items():
            s["success_rate"] = round(s["successes"] / s["tries"], 2) if s["tries"] else 0.0
        return stats

    def response_stats_for_family(self, dna_ids: List[str]) -> dict:
        """Aggregate response success rates across a whole matched threat
        family (used when the exact DNA is new but similar ones exist)."""
        if not dna_ids:
            return {}
        placeholders = ",".join("?" for _ in dna_ids)
        rows = self.conn.execute(
            f"SELECT action, success FROM responses WHERE dna_id IN ({placeholders})", dna_ids
        ).fetchall()
        stats: dict = {}
        for action, success in rows:
            s = stats.setdefault(action, {"tries": 0, "successes": 0})
            s["tries"] += 1
            s["successes"] += success
        for action, s in stats.items():
            s["success_rate"] = round(s["successes"] / s["tries"], 2) if s["tries"] else 0.0
        return stats

    # ---- incidents --------------------------------------------------------
    def store_incident(self, incident: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO incidents "
            "(incident_id, dna_id, anomaly_score, matched_dna_id, similarity, recommended_action, risk_level, status, source, scenario_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                incident["incident_id"], incident["dna_id"], incident["anomaly_score"],
                incident.get("matched_dna_id"), incident.get("similarity", 0.0),
                incident.get("recommended_action"), incident.get("risk_level"),
                incident.get("status", "pending"), incident.get("source", "synthetic_telemetry"),
                incident.get("scenario_id"), time.time(),
            ),
        )
        self.conn.commit()

    def update_incident_status(self, incident_id: str, status: str):
        self.conn.execute("UPDATE incidents SET status=? WHERE incident_id=?", (status, incident_id))
        self.conn.commit()

    def all_incidents(self) -> List[dict]:
        cols = ["incident_id", "dna_id", "anomaly_score", "matched_dna_id", "similarity",
                "recommended_action", "risk_level", "status", "source", "scenario_id", "created_at"]
        rows = self.conn.execute(
            f"SELECT {', '.join(cols)} FROM incidents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def get_incident(self, incident_id: str) -> Optional[dict]:
        cols = ["incident_id", "dna_id", "anomaly_score", "matched_dna_id", "similarity",
                "recommended_action", "risk_level", "status", "source", "scenario_id", "created_at"]
        row = self.conn.execute(
            f"SELECT {', '.join(cols)} FROM incidents WHERE incident_id=?", (incident_id,)
        ).fetchone()
        return dict(zip(cols, row)) if row else None


def dna_from_dict(d: dict) -> AttackDNA:
    """Rehydrate an AttackDNA dataclass from its stored dict form."""
    return AttackDNA(
        dna_id=d["dna_id"],
        attack_type=d["attack_type"],
        network=NetworkGenome(**d["network"]),
        behavior=BehavioralGenome(**d["behavior"]),
        resource=ResourceGenome(**d["resource"]),
        context=ContextGenome(**d["context"]),
        evolution=EvolutionMeta(**{k: v for k, v in d["evolution"].items() if k != "created_at"} |
                                 {"created_at": d["evolution"].get("created_at", time.time())}),
    )
