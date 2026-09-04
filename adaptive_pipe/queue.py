# -*- coding: utf-8 -*-
"""
Thin wrapper around teif_pipeline.review.store.ReviewStore -- NOT a new
review system. Same SQLite file, same schema, same corrections table the
NER pipeline would use too, so corrections recorded from either pipeline
build one shared per-seller memory (memory.py reads it back the same way
regardless of which pipeline originally submitted the document).

This module only adds the two things adaptive_pipe needs that the raw
store API doesn't give you directly: turning a verify.Problem list into
the store's gate_results_json shape, and opening the store at
config.REVIEW_DB_PATH by default.
"""

from __future__ import annotations

import json
import os

from teif_pipeline.models import Invoice
from teif_pipeline.review.store import ReviewStore

from . import config


def open_store(db_path: str | None = None) -> ReviewStore:
    path = db_path or config.REVIEW_DB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return ReviewStore(db_path=path)


def submit_for_review(
    store: ReviewStore, invoice: Invoice, source_text: str,
    problems: list, pdf_path: str | None = None, fields: dict | None = None,
) -> int:
    """Puts a document in the queue exactly like teif_pipeline's own
    review API does (see teif_pipeline/review/api.py's submit_invoice) --
    same tables, same shape, so a corrected field here is found by
    memory.py on the next document from this seller regardless of which
    pipeline reviews it.

    `gate_results_json` here holds an OBJECT, not the plain list
    teif_pipeline's own API writes -- {"problems": [...], "fields": {...}}
    -- so the review UI can show/highlight the same flat field keys
    extract.py and memory.py use, not just the nested Invoice shape. The
    column is untyped JSON text either way, and nothing in this repo reads
    it assuming one shape across both pipelines, so this is safe."""
    gate_results = {
        "problems": [
            {"gate": "adaptive_pipe.verify", "severity": p.severity,
             "field_name": p.field, "message": p.message}
            for p in problems
        ],
        "fields": fields or {},
    }
    return store.submit(
        source_text=source_text,
        invoice_json=invoice.model_dump_json(exclude={"source_text"}),
        gate_results_json=json.dumps(gate_results),
        source_path=pdf_path,
    )


def flat_fields_with_corrections(store: ReviewStore, item_id: int) -> dict:
    """The flat fields dict this item was submitted with, overlaid with
    whatever corrections have been recorded since -- so the UI (and any
    highlight lookup) always works off the current best-known value, not
    the stale original one."""
    item = store.get(item_id)
    if item is None:
        return {}
    try:
        gate_results = json.loads(item.gate_results_json)
        fields = dict(gate_results.get("fields") or {}) if isinstance(gate_results, dict) else {}
    except json.JSONDecodeError:
        fields = {}

    for c in store.corrections_for(item_id):
        flat_key = config.DOTTED_TO_FLAT.get(c["field_name"], c["field_name"])
        fields[flat_key] = c["corrected_value"]
    return fields


def problems_for(store: ReviewStore, item_id: int) -> list[dict]:
    item = store.get(item_id)
    if item is None:
        return []
    try:
        gate_results = json.loads(item.gate_results_json)
    except json.JSONDecodeError:
        return []
    if isinstance(gate_results, dict):
        return gate_results.get("problems") or []
    return gate_results if isinstance(gate_results, list) else []
