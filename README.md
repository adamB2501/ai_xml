# adaptive_pipe

PDF invoice → TEIF (Tunisian e-invoice) XML, using a local vision + text
LLM to read the document, deterministic checks to decide what's actually
trustworthy, a per-seller memory built from real corrections, and a
review UI for anything that isn't.

**Run this one:** `adaptive_pipe`. The other two packages are its
building blocks, not separate tools you invoke yourself.

| package | what it is |
|---|---|
| `adaptive_pipe/` | **The pipeline.** Tiered extraction (primary model, escalation model if configured), a hard gate (nothing reaches a filing-ready XML without passing every check or being approved by a human), per-seller memory from real corrections, and the review UI. |
| `accurate_pipe/` | The extraction machinery `adaptive_pipe` builds on: PDF → text (pdfplumber, plus a vision model for scanned pages / rasterized letterhead-footer bands) → structured fields (schema-constrained LLM call) → deterministic verification → targeted re-ask loop. |
| `teif_pipeline/` | The shared foundation: the `Invoice`/`Party`/`LineItem` data model, numeric/date parsing, the TEIF XML builder + validator, and the review queue's SQLite store. |

## Setup

```bash
pip install -r requirements.txt
```

Then install [Ollama](https://ollama.com) separately (not pip-installable)
and pull the two models this project calls:

```bash
ollama pull qwen2.5vl:7b     # vision -- scanned pages, rasterized letterhead/footer bands
ollama pull qwen3:14b        # text extraction (adaptive_pipe.config.TEXT_MODEL)
```

**Hardware note:** every model call in this project runs locally through
Ollama — no cloud API. On a machine with little or no usable VRAM,
Ollama falls back to CPU and a single extraction can take several
minutes; `qwen3:14b` in particular is the slow one (~9 minutes/call on
the CPU-only machine this was built on). With **2GB VRAM**, don't expect
a meaningful GPU speedup from either model as configured — `qwen2.5vl:7b`
needs ~6GB and `qwen3:14b` ~10-12GB to meaningfully offload, so both will
likely still run mostly/fully on CPU. If that's too slow in practice,
the cheap fix is smaller models, not more code:
`ollama pull qwen2.5vl:3b` and a smaller text model, then set
`ACCURATE_PIPE_VISION_MODEL` / `ADAPTIVE_PIPE`'s `ACCURATE_PIPE_TEXT_MODEL`
env vars accordingly (see `adaptive_pipe/config.py`). 16GB system RAM is
enough to hold one loaded quantized model at a time comfortably; Ollama
unloads a model after ~5 minutes idle by default.

## Run it

```bash
# the review UI (recommended) -- browse/pick files or folders, watch progress,
# review+correct extracted fields side by side with the PDF, approve
uvicorn adaptive_pipe.ui.server:app --reload
# open http://127.0.0.1:8000

# or, one document at a time from the command line
python -m adaptive_pipe.run path/to/invoice.pdf
```

See `adaptive_pipe/README.md` for the full design: the tiering/escalation
logic, the memory system's two paths (and why history is only ever a
candidate to verify, never a value to trust), and the review queue.

## Tests

```bash
pip install -r requirements.txt   # includes pytest + httpx
pytest -q
```

39 tests, no live Ollama server required — every model call is
monkeypatched with a scripted fake; the tests that touch the review
queue/UI use a real temp SQLite database and the real sample PDF in
`data/docs/facture.pdf`.
