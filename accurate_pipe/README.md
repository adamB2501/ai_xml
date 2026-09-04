# accurate_pipe

A second invoice → TEIF-XML pipeline, built for **messy real invoices**
(crooked scans, dropped words, PDFs whose text layer is only half there).

It is deliberately separate from `teif_pipeline/` (the spaCy-NER pipeline).
It **reuses** `teif_pipeline`'s canonical `Invoice` model, TEIF XML builder,
and XML validator — those are correct and tested. Only the *reading* half
is done differently here.

---

## The idea in one paragraph

Use the most reliable source for each part of the page: exact embedded text
where a real text layer exists (pdfplumber), and a **vision model**
(`qwen2.5vl` via Ollama) only for the parts that are just pixels — a
rasterized letterhead/footer band, or a fully scanned page. Then hand that
gathered text to a **text LLM** whose only job is to turn it into
structured fields, with a strict prompt **and a mandatory "re-check your
own answer" pass**. Then run **deterministic** cross-checks (arithmetic,
tax-ID shape, "does this value actually appear in the source"). Then build
and validate the XML. Anything that fails a check, or came from a scan,
is flagged **needs human review** — not auto-filed.

---

## Flow

```
PDF ─▶ ingest ─▶ extract ─▶ verify ─▶ reask ─▶ assemble ─▶ TEIF build ─▶ TEIF validate ─▶ result
        │          │          │         │         │
        │          │          │         │         └─ fields dict → teif_pipeline.models.Invoice
        │          │          │         └─ re-ask the model for ONLY the failing fields, with
        │          │          │            hints; accept a value only if it still passes the
        │          │          │            checks; cap retries per field; leftovers → review
        │          │          └─ deterministic checks → list[Problem]  (this gates the result)
        │          └─ text LLM + self-recheck + escalate-if-flagged → fields dict
        └─ classify {text|mixed|scanned}; pdfplumber + vision model as needed → source text
```

| classification | how the text is gathered | model calls |
|---|---|---|
| `text` | pdfplumber only | none |
| `mixed` | pdfplumber body **+** vision model on each rasterized band | 1 vision call per band |
| `scanned` | vision model transcribes every page | 1 vision call per page, result flagged low-confidence |

---

## Files (read in this order)

| file | what it is |
|---|---|
| `config.py` | every tunable value, each explained. Override via `ACCURATE_PIPE_*` env vars. |
| `prompts.py` | the exact text sent to the models, with a comment on **why** each rule exists. The self-recheck ("PASS 2") lives here. |
| `llm.py` | the only place that talks to Ollama (`/api/generate`, `/api/chat`). stdlib `urllib`, no deps. |
| `ingest.py` | STEP 1. PDF → source text. Classification + region geometry + vision transcription. |
| `numparse.py` | number/date parsing — re-exported from `teif_pipeline.numeric`, not re-implemented. |
| `extract.py` | STEP 2. source text → fields dict. One text-LLM call + escalation logic. |
| `verify.py` | STEP 3. deterministic checks → `list[Problem]` (`error` blocks, `warning` informs). |
| `reask.py` | STEP 3b. loop: re-ask the model for only the fields that failed, with hints; accept only if the value passes the checks; `REASK_MAX_ATTEMPTS_PER_FIELD` tries each, then → review. |
| `assemble.py` | STEP 4a. fields dict → `teif_pipeline.models.Invoice`. |
| `pipeline.py` | the four steps wired together → `PipelineResult`. |
| `run.py` | CLI wrapper. Writes `<name>.fields.json`, `<name>.xml`, `<name>.report.txt`. |

---

## Run it

```bash
# one invoice
python -m accurate_pipe.run data/docs/facture.pdf

# custom output dir, no stdout dump
python -m accurate_pipe.run data/docs/facture.pdf --out /tmp/x --quiet

# try the heavier text model for one run
ACCURATE_PIPE_TEXT_MODEL=qwen3:14b python -m accurate_pipe.run data/docs/facture.pdf
```

Exit codes: `0` = XML produced, no review needed · `2` = XML produced but
**needs human review** · `1` = the run failed.

### Prerequisites

- `ollama serve` running locally
- `ollama pull qwen2.5vl:7b`  (vision)
- `ollama pull mistral:latest`  and/or  `ollama pull qwen3:14b`  (text)
- Python deps: `pdfplumber`, `pillow`, `pydantic` (already in the project);
  everything else is stdlib.

> On a CPU-only machine, a full-page vision call is minutes, not seconds.
> `text`-classified PDFs skip the vision model entirely.

---

## What this pipeline does NOT do

- It does not correct values silently. A failed check is reported; fixing
  it is a human's job (or a future targeted re-ask).
- It does not re-implement TEIF. XML shape, referential codes, and
  validation come from `teif_pipeline`.
- It does not decide a document is "fine" on the model's word alone — the
  deterministic checks in `verify.py` are the gate.
