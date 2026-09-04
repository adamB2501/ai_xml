# -*- coding: utf-8 -*-
"""
The review UI's backend. Run with:

    uvicorn adaptive_pipe.ui.server:app --reload

then open http://127.0.0.1:8000 . Everything here either (a) reads/writes
the real teif_pipeline.review SQLite store via adaptive_pipe.queue, or
(b) kicks off adaptive_pipe.pipeline.run() through jobs.py. No separate
state, no new database.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from accurate_pipe import assemble, prompts
from teif_pipeline.teif.builder import build_teif_xml

from . import highlight, jobs, queue_bridge
from .. import config as ap_config
from .. import queue as ap_queue

app = FastAPI(title="adaptive_pipe review")

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# color palette -- same field, same color, everywhere it's shown
# ---------------------------------------------------------------------------

PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9A6324", "#469990", "#800000",
    "#808000", "#000075", "#e6beff", "#fabed4", "#aaffc3", "#ffd8b1",
]
_FIELD_ORDER = [k for k, _ in prompts.FIELD_SPEC]


def color_for(key: str) -> str:
    m = re.match(r"line_items\[(\d+)\]", key)
    if m:
        return PALETTE[(int(m.group(1)) + 8) % len(PALETTE)]
    try:
        idx = _FIELD_ORDER.index(key)
    except ValueError:
        idx = abs(hash(key))
    return PALETTE[idx % len(PALETTE)]


# ---------------------------------------------------------------------------
# static page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/field-spec")
def field_spec():
    """The canonical field list (accurate_pipe.prompts.FIELD_SPEC /
    LINE_ITEM_SPEC) -- single source of truth, so the review screen always
    shows EVERY field the pipeline knows how to extract, not just the ones
    a given document happened to produce. That's what makes a completely
    omitted field visible and fillable-in, instead of just invisible."""
    return {
        "fields": [{"key": k, "label": d} for k, d in prompts.FIELD_SPEC],
        "line_item_fields": [{"key": k, "label": d} for k, d in prompts.LINE_ITEM_SPEC],
    }


# ---------------------------------------------------------------------------
# folder / file browsing + batch processing
# ---------------------------------------------------------------------------

_dpi_awareness_set = False


def _ensure_dpi_aware() -> None:
    """Without this, Windows treats the process as DPI-unaware and
    bitmap-stretches every window it opens to match the display scale --
    that's the actual cause of the native dialog looking pixelated on a
    HiDPI screen, not a tkinter rendering bug. Telling Windows this
    process handles its own scaling makes the dialog render at native
    resolution instead. Idempotent and safe to call repeatedly (the
    module-level flag avoids a second call, which can raise on some
    Windows versions once awareness is already set); silently a no-op on
    non-Windows or if the API isn't available."""
    global _dpi_awareness_set
    if _dpi_awareness_set:
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        pass
    _dpi_awareness_set = True


def _native_pick_files() -> list[str]:
    """Opens the REAL Windows file-open dialog (tkinter's filedialog wraps
    the native common dialog -- this is not a custom in-page widget) and
    blocks until the user picks PDF(s) or cancels. Runs on whatever thread
    FastAPI's threadpool gives this sync endpoint, which is exactly what
    you want here -- the event loop keeps serving other requests (queue
    polling, etc.) while the dialog is open."""
    _ensure_dpi_aware()
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        paths = filedialog.askopenfilenames(
            title="Select invoice PDF(s)", filetypes=[("PDF files", "*.pdf")]
        )
    finally:
        root.destroy()
    return list(paths)


def _native_pick_folder() -> str | None:
    _ensure_dpi_aware()
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        folder = filedialog.askdirectory(title="Select a folder of invoices")
    finally:
        root.destroy()
    return folder or None


@app.post("/api/pick-files")
def pick_files():
    """Native OS file-open dialog -> the PDFs the user selected."""
    try:
        paths = _native_pick_files()
    except Exception as exc:  # noqa: BLE001 - e.g. no display available
        raise HTTPException(500, f"couldn't open the native file dialog: {exc}")
    return {"paths": paths}


@app.post("/api/pick-folder")
def pick_folder():
    """Native OS folder dialog -> every PDF found under it (recursive), so
    picking a folder full of invoices needs exactly one dialog, not a
    second in-page browse step."""
    try:
        folder = _native_pick_folder()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"couldn't open the native folder dialog: {exc}")
    if not folder:
        return {"folder": None, "paths": []}
    pdfs = sorted(str(p) for p in Path(folder).rglob("*.pdf"))
    return {"folder": folder, "paths": pdfs}


@app.get("/api/browse")
def browse(path: Optional[str] = None):
    """Kept as a fallback for a headless/no-display environment where the
    native dialogs above can't open -- the UI itself now drives file/
    folder selection through pick-files/pick-folder instead of this."""
    target = Path(path) if path else Path.cwd()
    if not target.exists() or not target.is_dir():
        raise HTTPException(400, f"not a directory: {target}")
    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            is_pdf = child.is_file() and child.suffix.lower() == ".pdf"
            if child.is_dir() or is_pdf:
                entries.append({"name": child.name, "path": str(child), "is_dir": child.is_dir()})
    except PermissionError:
        raise HTTPException(403, f"cannot list: {target}")
    return {"path": str(target), "parent": str(target.parent), "entries": entries}


class ProcessRequest(BaseModel):
    paths: list[str]


@app.post("/api/process")
def process(req: ProcessRequest):
    missing = [p for p in req.paths if not os.path.isfile(p)]
    if missing:
        raise HTTPException(400, f"not found: {missing}")
    job = jobs.start_batch(req.paths)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return [j.to_dict() for j in jobs.list_batches()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_batch(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


# ---------------------------------------------------------------------------
# review queue
# ---------------------------------------------------------------------------

def _store():
    return queue_bridge.get_store()


@app.get("/api/queue")
def list_queue(status: Optional[str] = None):
    items = _store().list_items(status=status)
    return [
        {"id": i.id, "source_path": i.source_path, "status": i.status, "created_at": i.created_at}
        for i in items
    ]


@app.get("/api/queue/{item_id}")
def get_queue_item(item_id: int):
    store = _store()
    item = store.get(item_id)
    if item is None:
        raise HTTPException(404, "not found")
    fields = ap_queue.flat_fields_with_corrections(store, item_id)
    problems = ap_queue.problems_for(store, item_id)
    return {
        "id": item.id,
        "source_path": item.source_path,
        "status": item.status,
        "created_at": item.created_at,
        "fields": fields,
        "problems": problems,
        "corrections": store.corrections_for(item_id),
    }


@app.get("/api/queue/{item_id}/pdf")
def get_queue_pdf(item_id: int):
    item = _store().get(item_id)
    if item is None or not item.source_path:
        raise HTTPException(404, "no source PDF on file for this item")
    if not os.path.isfile(item.source_path):
        raise HTTPException(404, f"source PDF no longer at {item.source_path}")
    return FileResponse(item.source_path, media_type="application/pdf")


@app.get("/api/queue/{item_id}/highlights")
def get_highlights(item_id: int):
    store = _store()
    item = store.get(item_id)
    if item is None:
        raise HTTPException(404, "not found")
    fields = ap_queue.flat_fields_with_corrections(store, item_id)
    if not item.source_path or not os.path.isfile(item.source_path):
        return {}
    rects = highlight.rects_for_fields(item.source_path, fields)
    return {
        key: {"color": color_for(key), "rects": r}
        for key, r in rects.items()
    }


class CorrectionRequest(BaseModel):
    field_name: str  # flat key, e.g. "buyer_tax_id" or "line_items[0].description"
    corrected_value: str


@app.post("/api/queue/{item_id}/correct")
def correct_field(item_id: int, req: CorrectionRequest):
    store = _store()
    item = store.get(item_id)
    if item is None:
        raise HTTPException(404, "not found")

    dotted = ap_config.FLAT_TO_DOTTED.get(req.field_name, req.field_name)
    fields_before = ap_queue.flat_fields_with_corrections(store, item_id)
    original_value = fields_before.get(req.field_name)

    store.correct_field(item_id, dotted, original_value, req.corrected_value)
    return {"status": "recorded", "field_name": req.field_name, "corrected_value": req.corrected_value}


@app.post("/api/queue/{item_id}/approve")
def approve(item_id: int):
    store = _store()
    if store.get(item_id) is None:
        raise HTTPException(404, "not found")
    store.approve(item_id)
    return {"status": "approved"}


@app.delete("/api/queue/{item_id}")
def delete_queue_item(item_id: int):
    """Removes an item from the review queue. Note: also removes whatever
    memory.py would have learned from this item's corrections for future
    documents from the same seller -- see ReviewStore.delete()'s docstring."""
    store = _store()
    if store.get(item_id) is None:
        raise HTTPException(404, "not found")
    store.delete(item_id)
    return {"status": "deleted"}


@app.get("/api/queue/{item_id}/xml")
def get_xml(item_id: int):
    store = _store()
    item = store.get(item_id)
    if item is None:
        raise HTTPException(404, "not found")
    fields = ap_queue.flat_fields_with_corrections(store, item_id)
    invoice = assemble.to_invoice(
        fields, source_text=item.source_text, model_used="adaptive_pipe:reviewed",
    )
    xml, warnings = build_teif_xml(invoice)
    return {"xml": xml, "warnings": warnings}
