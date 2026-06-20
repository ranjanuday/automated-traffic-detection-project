"""SQLite persistence for processed images & violations.

Two tables: `evidence` (one row per processed image) and `violation` (one row
per detected violation, FK to evidence). Plain stdlib sqlite3 — zero extra
deps, perfect for a prototype. Plate numbers are PII, so the DB lives under
data/ which is git-ignored.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    original_path TEXT NOT NULL,
    annotated_path TEXT NOT NULL,
    n_objects     INTEGER NOT NULL,
    n_violations  INTEGER NOT NULL,
    backend       TEXT,
    metrics_json  TEXT
);
CREATE TABLE IF NOT EXISTS violation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id INTEGER NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    vtype       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    plate       TEXT,
    note        TEXT,
    bbox_json   TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_violation_type ON violation(vtype);
CREATE INDEX IF NOT EXISTS idx_violation_plate ON violation(plate);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_result(original_path: str, annotated_path: str, frame, metrics: dict) -> int:
    """Persist one processed image + its violations. Returns evidence id."""
    ts = _now()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO evidence
               (created_at, original_path, annotated_path, n_objects,
                n_violations, backend, metrics_json)
               VALUES (?,?,?,?,?,?,?)""",
            (ts, original_path, annotated_path, len(frame.detections),
             len(frame.violations), metrics.get("detector_backend"),
             json.dumps(metrics)),
        )
        eid = cur.lastrowid
        for v in frame.violations:
            c.execute(
                """INSERT INTO violation
                   (evidence_id, vtype, confidence, plate, note, bbox_json,
                    created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (eid, v.vtype, v.confidence, v.plate, v.note,
                 json.dumps(list(v.bbox)), ts),
            )
        return eid


def list_evidence(limit: int = 100, vtype: str = "", plate: str = "") -> list[dict]:
    """Searchable records — filter by violation type and/or plate."""
    q = """SELECT e.*,
                  (SELECT GROUP_CONCAT(DISTINCT v2.vtype)
                   FROM violation v2 WHERE v2.evidence_id = e.id) AS vtypes
           FROM evidence e"""
    where, args = [], []
    if vtype:
        where.append("EXISTS (SELECT 1 FROM violation v WHERE v.evidence_id=e.id AND v.vtype=?)")
        args.append(vtype)
    if plate:
        where.append("EXISTS (SELECT 1 FROM violation v WHERE v.evidence_id=e.id AND v.plate LIKE ?)")
        args.append(f"%{plate.upper()}%")
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY e.id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def get_evidence(eid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM evidence WHERE id=?", (eid,)).fetchone()
        if not row:
            return None
        ev = dict(row)
        ev["violations"] = [
            dict(r) for r in c.execute(
                "SELECT * FROM violation WHERE evidence_id=? ORDER BY confidence DESC",
                (eid,)).fetchall()
        ]
        return ev


def stats() -> dict[str, Any]:
    """Aggregates for the analytics dashboard."""
    with _conn() as c:
        totals = c.execute(
            "SELECT COUNT(*) n_ev, COALESCE(SUM(n_violations),0) n_vi FROM evidence"
        ).fetchone()
        by_type = c.execute(
            "SELECT vtype, COUNT(*) n FROM violation GROUP BY vtype ORDER BY n DESC"
        ).fetchall()
        by_day = c.execute(
            """SELECT substr(created_at,1,10) d, COUNT(*) n
               FROM violation GROUP BY d ORDER BY d"""
        ).fetchall()
        top_plates = c.execute(
            """SELECT plate, COUNT(*) n FROM violation
               WHERE plate IS NOT NULL GROUP BY plate ORDER BY n DESC LIMIT 10"""
        ).fetchall()
    return {
        "n_evidence": totals["n_ev"],
        "n_violations": totals["n_vi"],
        "by_type": {r["vtype"]: r["n"] for r in by_type},
        "by_day": {r["d"]: r["n"] for r in by_day},
        "top_plates": [{"plate": r["plate"], "count": r["n"]} for r in top_plates],
    }
