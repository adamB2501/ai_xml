# adaptive_pipe

The third pipeline, built directly out of a long back-and-forth about what
was actually wrong with the second one (`accurate_pipe`). Short version:
`accurate_pipe`'s extraction was fine — the problems were (1) it escalated
to a heavier model based on the model's own unreliable opinion of itself
instead of an actual deterministic check, and (2) every invoice from a
seller you've seen ten times before is treated exactly like the first one
ever, with zero memory. This pipeline fixes both **without re-implementing
anything that already worked** — it imports `accurate_pipe`'s ingest,
prompts, llm, verify, assemble, and (extended, not rewritten) re-ask
modules directly.

## The one rule everything else follows

**A past correction is a candidate to test, never a value to trust.**
Nothing here ever writes a field into the final record because "we saw it
before." Every value that reaches the final Invoice was either found and
checked against *this specific document's* own text, or a human confirmed
it. See `memory.py`'s docstring — this rule is the entire reason that
module is shaped the way it is, and it exists because an earlier version
of this plan assumed things like a seller's bank details were "constants"
and got (correctly) called out for it: nothing is assumed constant here,
things are *re-confirmed*, every time.

## What's actually new here (vs. accurate_pipe)

1. **`memory.py`** — reads `teif_pipeline.review`'s real correction log
   (SQLite, already built, previously unused) for the current document's
   seller. Two paths:
   - **Direct reuse** (no model call): for fields that are genuinely
     stable per seller (name, address, RC number, capital), if a
     previously-corrected value is found *verbatim in this document's own
     text*, it's used. Not found → not used, falls back to normal
     extraction. Never injected blind.
   - **Positional hints** (still a model call): for fields that vary per
     invoice (buyer tax id, amounts, dates), history can't supply the
     value, but it can say *where things like it tend to sit* relative to
     a label on this seller's invoices — merged into the same re-ask
     prompt `accurate_pipe.reask` already builds.
2. **`queue.py`** — submits an unresolved document to `teif_pipeline
   .review`'s real SQLite queue (not a new review system — the same
   tables the NER pipeline's review API already uses, so a correction
   made through either pipeline is visible to both).
3. **`pipeline.py`** — the escalation fix: light model → verify → memory
   + re-ask → **if `verify.py`'s deterministic result still says it's
   broken** (not the model's self-report) → heavy model, same process →
   still broken → submit for review, **no filing-ready XML is produced**.
   Clean at either tier → build + validate immediately, no human touch.

## Flow

```
PDF -> accurate_pipe.ingest -> source text
    -> extract.extract_once(LIGHT model)         -> fields
    -> memory.apply_direct_reuse                 -> fields   (Path 1)
    -> accurate_pipe.verify.verify               -> problems
    -> accurate_pipe.reask.run(LIGHT model,
           extra_hints=memory.hints_from_history) -> fields, problems  (Path 2)
    -> still failing (verify's answer, not the model's)?
         no  -> assemble -> build TEIF XML -> validate -> done
         yes + a heavier model is configured
             -> repeat the whole block with the HEAVY model
             -> still failing? -> queue.submit_for_review -> STOP (no XML)
```

## Run it

```bash
python -m adaptive_pipe.run data/docs/facture.pdf
```

Writes `<name>.fields.json` and `<name>.report.txt` always;
`<name>.xml` **only if `needs_review` is False** — see `pipeline.py`'s
docstring for why a flagged document doesn't get a filing-ready XML at
all, rather than an XML marked "draft."

The review queue is a real SQLite file at
`data/output/adaptive_pipe/review_queue.db` (configurable via
`ADAPTIVE_PIPE_REVIEW_DB`). Correct a field with the existing
`teif_pipeline.review` API/store directly:

```python
from teif_pipeline.review.store import ReviewStore
store = ReviewStore("data/output/adaptive_pipe/review_queue.db")
store.correct_field(item_id, "buyer.tax_id", original_value=None, corrected_value="335352FAP000")
```

That correction is immediately visible to `memory.py` on the *next*
document from the same seller (matched by `seller_tax_id` — the one field
that both identifies a seller and is always freshly re-extracted, never
assumed).

## Files

| file | what's in it |
|---|---|
| `config.py` | re-exports the accurate_pipe settings this pipeline needs, plus what's new: review DB path, which fields count as "stable per seller" (`STABLE_SELLER_FIELDS`), the flat-key ↔ dotted-path field name table, hint confidence threshold. |
| `memory.py` | the per-seller memory — read the log, `apply_direct_reuse` (Path 1), `hints_from_history` (Path 2). Read this one first. |
| `queue.py` | thin wrapper around `teif_pipeline.review.store.ReviewStore`. |
| `extract.py` | one model, one extraction call, no escalation logic (that decision moved to `pipeline.py`, where it can see `verify.py`'s actual result). |
| `pipeline.py` | the tiering, wired together. |
| `run.py` | CLI. |

## What this does *not* claim to do

- It does not guarantee a wrong value never reaches the XML — no system
  built on an LLM/VLM can promise that. What it guarantees is narrower and
  actually achievable: nothing reaches a filing-ready XML without being
  independently corroborated against the current document or approved by
  a person.
- The memory layer helps a specific seller *after* at least one of their
  invoices has been through review once. It does nothing on the very
  first invoice from a brand-new seller — there's no history to draw on
  yet, and that's expected, not a bug.
- It is not a fourth rewrite of extraction logic. If `accurate_pipe`'s
  ingest/verify/reask behavior changes, this pipeline changes with it —
  that's deliberate, not incidental.
