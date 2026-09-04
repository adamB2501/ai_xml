# -*- coding: utf-8 -*-
"""One process-wide ReviewStore handle, shared by server.py's endpoints
and jobs.py's background threads, so both see the same SQLite connection
target. A single module-level function (not a bare global) so tests can
swap it to a temp-file store the same way tests/unit/test_review_api.py
does for teif_pipeline's own API."""

from __future__ import annotations

from .. import queue as ap_queue

_store = None


def get_store():
    global _store
    if _store is None:
        _store = ap_queue.open_store()
    return _store


def set_store(store) -> None:
    global _store
    _store = store
