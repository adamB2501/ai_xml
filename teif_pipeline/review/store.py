# -*- coding: utf-8 -*-
"""Persistent store for the review queue (Phase 6), SQLite-backed (stdlib
sqlite3 -- no new dependency for something this project-scale). Two
tables: `review_items` (one row per document that needs or received human
attention) and `corrections` (one row per field a human corrected).

The `corrections` table is the actual point of this phase, per the brief:
"Every human correction is stored as a labeled example on a real document.
This is how the project acquires real training data over time." -- i.e.
the thing this whole project has been missing (see docs/phase0_audit.md,
and every conversation this session about having zero real labeled
invoices). Each row here has enough to become a real training example:
source_text + field_name + corrected_value, on a genuine document instead
of another synthetic one.
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT,
    source_text TEXT NOT NULL,
    invoice_json TEXT NOT NULL,
    gate_results_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | corrected
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_item_id INTEGER NOT NULL REFERENCES review_items(id),
    field_name TEXT NOT NULL,
    original_value TEXT,
    corrected_value TEXT NOT NULL,
    corrected_at TEXT NOT NULL
);
"""


@dataclass
class ReviewItem:
    id: Optional[int]
    source_path: Optional[str]
    source_text: str
    invoice_json: str
    gate_results_json: str
    status: str
    created_at: str


class ReviewStore:
    def __init__(self, db_path: str = "review_queue.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True) if Path(db_path).parent != Path(".") else None
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def submit(self, source_text: str, invoice_json: str, gate_results_json: str, source_path: Optional[str] = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO review_items (source_path, source_text, invoice_json, gate_results_json, status, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (source_path, source_text, invoice_json, gate_results_json, now),
            )
            return cur.lastrowid

    def get(self, item_id: int) -> Optional[ReviewItem]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM review_items WHERE id = ?", (item_id,)).fetchone()
        return ReviewItem(**dict(row)) if row else None

    def list_items(self, status: Optional[str] = None) -> list[ReviewItem]:
        with self._connect() as conn:
            if status:
                rows = conn.execute("SELECT * FROM review_items WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM review_items ORDER BY id DESC").fetchall()
        return [ReviewItem(**dict(r)) for r in rows]

    def approve(self, item_id: int):
        with self._connect() as conn:
            conn.execute("UPDATE review_items SET status = 'approved' WHERE id = ?", (item_id,))

    def delete(self, item_id: int) -> None:
        """Removes a review item and its corrections. SQLite doesn't
        cascade-delete here (the schema has no ON DELETE CASCADE), so both
        tables are cleared explicitly.

        Note this also removes whatever this item's corrections were
        contributing to adaptive_pipe.memory's per-seller hints -- a
        correction only exists in this table, so deleting it here means
        future documents from the same seller lose that signal too, not
        just this one item. Delete a genuinely wrong/duplicate/unwanted
        item; don't delete something just to tidy the queue if its
        correction was real and useful."""
        with self._connect() as conn:
            conn.execute("DELETE FROM corrections WHERE review_item_id = ?", (item_id,))
            conn.execute("DELETE FROM review_items WHERE id = ?", (item_id,))

    def correct_field(self, item_id: int, field_name: str, original_value, corrected_value: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO corrections (review_item_id, field_name, original_value, corrected_value, corrected_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (item_id, field_name, json.dumps(original_value), corrected_value, now),
            )
            conn.execute("UPDATE review_items SET status = 'corrected' WHERE id = ?", (item_id,))

    def corrections_for(self, item_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM corrections WHERE review_item_id = ? ORDER BY id", (item_id,)).fetchall()
        return [dict(r) for r in rows]

    def all_corrections(self) -> list[dict]:
        """Every human correction ever recorded, joined with the source
        text it applies to -- this is the export path to turn accumulated
        corrections into real training examples."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.*, r.source_text FROM corrections c JOIN review_items r ON c.review_item_id = r.id ORDER BY c.id"
            ).fetchall()
        return [dict(r) for r in rows]
