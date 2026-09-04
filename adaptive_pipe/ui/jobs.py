# -*- coding: utf-8 -*-
"""
Minimal in-memory background job runner. A single document can take
several minutes on CPU (vision + text model calls) -- this exists so
"process this folder" returns immediately and the UI polls progress,
instead of an HTTP request hanging for however long N documents take.

No task-queue framework: this is a single-user local tool, so a dict
guarded by a lock plus one thread per batch is enough, and it adds zero
new dependencies. Jobs live only for the life of the server process --
restarting it clears history, which is fine, since the actual result of
each run (the review queue entry / the written XML) already persisted to
disk before the job object would be needed again.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from .. import pipeline as ap_pipeline
from . import queue_bridge

_OUTPUT_DIR = "data/output/adaptive_pipe"


@dataclass
class FileJob:
    path: str
    status: str = "queued"        # queued | running | done | failed
    phase: str = "Waiting…"       # human-readable current step, from pipeline.run's on_progress
    phase_seq: int = 0            # increments on every phase change -- lets the UI notice movement
                                  # even though we deliberately don't fake a completion percentage
                                  # (a single phase can be one multi-minute model call; there's no
                                  # honest sub-progress within it, see adaptive_pipe/README.md)
    error: str | None = None
    needs_review: bool | None = None
    review_item_id: int | None = None
    tier_used: str | None = None
    xml_path: str | None = None   # set only when needs_review is False -- see _run_batch
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path, "status": self.status, "phase": self.phase,
            "phase_seq": self.phase_seq, "error": self.error,
            "needs_review": self.needs_review, "review_item_id": self.review_item_id,
            "tier_used": self.tier_used, "xml_path": self.xml_path,
        }


@dataclass
class BatchJob:
    id: str
    files: list[FileJob]
    status: str = "queued"        # queued | running | done

    def to_dict(self) -> dict:
        done = sum(1 for f in self.files if f.status in ("done", "failed"))
        return {
            "id": self.id, "status": self.status,
            "progress": {"done": done, "total": len(self.files)},
            "files": [f.to_dict() for f in self.files],
        }


_JOBS: dict[str, BatchJob] = {}
_LOCK = threading.Lock()


def start_batch(paths: list[str]) -> BatchJob:
    job = BatchJob(id=uuid.uuid4().hex[:12], files=[FileJob(path=p) for p in paths])
    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run_batch, args=(job,), daemon=True).start()
    return job


def get_batch(job_id: str) -> BatchJob | None:
    with _LOCK:
        return _JOBS.get(job_id)


def list_batches() -> list[BatchJob]:
    with _LOCK:
        return sorted(_JOBS.values(), key=lambda j: j.id, reverse=True)


def _run_batch(job: BatchJob) -> None:
    job.status = "running"
    store = queue_bridge.get_store()
    for f in job.files:
        f.status = "running"
        f.started_at = time.time()

        def _report(phase: str, _f=f) -> None:
            _f.phase = phase
            _f.phase_seq += 1

        try:
            result = ap_pipeline.run(f.path, store=store, on_progress=_report)
            f.needs_review = result.needs_review
            f.review_item_id = result.review_item_id
            f.tier_used = result.tier_used
            if not result.needs_review and result.xml:
                # a clean pass produces a real XML string that otherwise has
                # nowhere to land when driven from this batch UI (unlike the
                # CLI in run.py, which already writes it) -- save it so a
                # successful run isn't silently discarded.
                os.makedirs(_OUTPUT_DIR, exist_ok=True)
                stem = os.path.splitext(os.path.basename(f.path))[0]
                out_path = os.path.join(_OUTPUT_DIR, f"{stem}.xml")
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(result.xml)
                f.xml_path = out_path
            f.status = "done"
            f.phase = "Done."
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the batch
            f.status = "failed"
            f.phase = "Failed."
            f.error = f"{type(exc).__name__}: {exc}\n" + "".join(
                traceback.format_exception(exc, limit=3)
            )
        f.finished_at = time.time()
    job.status = "done"
