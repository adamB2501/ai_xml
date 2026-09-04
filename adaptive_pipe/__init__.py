# -*- coding: utf-8 -*-
"""
adaptive_pipe
=============

The pipeline that came out of the "resonate, don't build" discussion:
accurate_pipe's extraction machinery (ingest / verify / re-ask), PLUS a
per-seller memory built from real human corrections, PLUS a hard gate that
nothing reaches a final, filing-ready XML without either passing every
deterministic check or being approved by a person.

It does NOT re-implement accurate_pipe. It imports its ingest, llm,
prompts, verify, assemble and numparse modules directly -- those already
work and are tested; rewriting them again would be exactly the "fourth
architecture" this was explicitly trying to avoid. What's new here is:

  1. memory.py    -- reads teif_pipeline.review's correction log for the
                     current document's seller and offers two things:
                       PATH 1 (no model call): if a past-corrected value
                       for a field that's actually stable per seller
                       (name, address, RC, capital) reappears verbatim in
                       THIS document's text, use it -- checked fresh
                       every time, never injected blindly.
                       PATH 2 (still a model call): for fields that vary
                       per invoice (buyer tax id, amounts, ...), a past
                       correction can't supply the value, but it can show
                       WHERE similar values tend to sit -- that becomes a
                       sharper hint fed into the existing re-ask prompt.
  2. queue.py     -- submits a document to teif_pipeline.review's real
                     SQLite queue when it can't be resolved automatically.
                     Not a new review system -- the same one, reused.
  3. pipeline.py  -- the tiering the old escalation trigger got wrong:
                     primary model -> verify -> memory + re-ask (capped at
                     REASK_MAX_ROUNDS calls) -> STILL failing (a fact, not
                     the model's own opinion) -> escalation model, if one
                     is configured (off by default -- see config.py) ->
                     verify -> memory + re-ask -> STILL failing -> submit
                     for review, do not build a filing-ready XML. Passes
                     clean at either tier -> build + validate immediately,
                     no human touch.

The invariant this whole thing exists to protect, stated once: no field
value is ever written to the final Invoice without being checked against
THIS document's own text (or arithmetic derived from THIS document's own
other fields). History and prior corrections are candidates to test, never
values to trust. See memory.py's docstring for why that distinction is
the entire point.

Entry point:  python -m adaptive_pipe.run path/to/invoice.pdf
"""

__all__ = ["pipeline", "config", "memory", "queue"]
