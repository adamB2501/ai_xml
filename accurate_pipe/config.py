# -*- coding: utf-8 -*-
"""
Every tunable value the pipeline uses, in one place, each with a note on
what it does and what happens if you change it. Nothing else in the
package hard-codes a threshold or a model name -- they all read from here.

You can override any of these without editing the file, via environment
variables (see the `_env*` helpers at the bottom). That's handy for
trying a bigger model on one run without touching code:

    ACCURATE_PIPE_TEXT_MODEL=qwen3:14b  python -m accurate_pipe.run x.pdf
"""

import os

# ---------------------------------------------------------------------------
# 1. Ollama connection
# ---------------------------------------------------------------------------

# Where the local Ollama server listens. Ollama's default; only change it
# if you started `ollama serve` with a custom OLLAMA_HOST.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# How long (seconds) to wait for one model response before giving up.
# On CPU, a full-page vision call can take several minutes -- keep this
# generous. A hang here means Ollama is stuck, not that the value is wrong.
OLLAMA_TIMEOUT_S = int(os.environ.get("ACCURATE_PIPE_TIMEOUT", "1800"))


# ---------------------------------------------------------------------------
# 2. Which models to use
# ---------------------------------------------------------------------------

# The VISION model: reads pixels (scanned pages, rasterized letterhead /
# footer bands). Must be a model Ollama tags as vision-capable.
#   qwen2.5vl:3b  - fastest, least accurate
#   qwen2.5vl:7b  - the default; good on clean scans, shaky on bad ones
#   qwen2.5vl:32b - noticeably better on degraded input, much slower / heavier
VISION_MODEL = os.environ.get("ACCURATE_PIPE_VISION_MODEL", "qwen2.5vl:7b")

# The TEXT model: turns the gathered text into structured fields. This is
# NOT a vision model -- it only ever sees text.
#   mistral:latest (7B) - the "lighter" option, ~2x faster on CPU
#   qwen3:14b           - the "heavier" option, better at the fiddly
#                         disambiguation (seller vs buyer tax id, scrambled
#                         columns) and at actually doing the recheck pass
#
# Pick with `scripts/compare_text_models.py` on your own invoices, then set
# whichever wins here. Default is the lighter one -- start cheap, escalate
# only if its accuracy isn't good enough for your documents.
TEXT_MODEL = os.environ.get("ACCURATE_PIPE_TEXT_MODEL", "mistral:latest")

# If the lighter text model's self-check reports unresolved problems, retry
# the SAME document once with this heavier model. Set to None to disable
# the escalation and always keep the light model's answer.
TEXT_MODEL_ESCALATION = os.environ.get("ACCURATE_PIPE_TEXT_MODEL_ESCALATION", "qwen3:14b") or None

# Sampling temperature. 0 = as deterministic as the model gets. Do not
# raise this for an extraction task -- you want the same answer every run.
TEMPERATURE = 0.0

# Context window (tokens) to ask Ollama to allocate. A full-page invoice
# image is ~4000 vision tokens; the text prompt + a page of invoice text
# is ~2000. 8192 leaves headroom for the model's own reasoning/recheck.
# Too small = the model silently loses the start of the document.
NUM_CTX = int(os.environ.get("ACCURATE_PIPE_NUM_CTX", "8192"))


# ---------------------------------------------------------------------------
# 3. PDF -> image rendering
# ---------------------------------------------------------------------------

# DPI for rasterizing a PDF page (or a region of it) before sending to the
# vision model. 300 is print quality. Higher = sharper but more pixels =
# more vision tokens = slower, with diminishing accuracy gains.
RENDER_DPI = int(os.environ.get("ACCURATE_PIPE_DPI", "300"))

# After rendering, downscale so the longer side is at most this many
# pixels. Qwen2.5-VL tiles large images internally anyway; ~1600 keeps
# text legible while cutting token count (and CPU time) substantially.
MAX_IMAGE_PX = int(os.environ.get("ACCURATE_PIPE_MAX_PX", "1600"))


# ---------------------------------------------------------------------------
# 4. Page classification: text / scanned / mixed
# ---------------------------------------------------------------------------
# (the logic lives in ingest.py; these are its cutoffs)

# Fewer than this many embedded characters on the whole document -> treat
# it as fully scanned (send every page image to the vision model).
SCANNED_CHAR_THRESHOLD = 20

# An image region counts as a "raster text band" worth OCRing when it:
#   - contains at most this many embedded characters (i.e. its text, if
#     any, is NOT already in the text layer)
MAX_CHARS_OVER_REGION = 3
#   - covers at least this fraction of the page area, OR ...
REGION_MIN_AREA_FRAC = 0.04
#   - ... spans at least this fraction of the page width (a header/footer
#     band is short but full-width)
REGION_MIN_WIDTH_FRAC = 0.5
#   - and is NOT essentially the whole page (that's a background image with
#     text drawn over it, or a scan -- handled elsewhere)
REGION_MAX_AREA_FRAC = 0.6

# A single image covering >= this fraction of the page, with text on top of
# it, means "scanned page that happens to carry an OCR text layer" -- the
# text is usable but scan-quality, so we flag the result low-confidence.
FULL_PAGE_IMAGE_FRAC = 0.85


# ---------------------------------------------------------------------------
# 4b. Targeted re-ask loop (reask.py)
# ---------------------------------------------------------------------------
# After the first extraction, verify.py says which fields are missing or
# failed a check. reask.py then asks the text model again for ONLY those
# fields, with hints (what's already known, arithmetically-implied values).
# A re-asked value is accepted only if it (a) appears verbatim in the
# source text and (b) does not make any gate worse - same rule as
# teif_pipeline's HybridBackend.

# How many times a single field may be re-asked before we give up on it
# and leave it for the human. Counted PER FIELD: a field that keeps
# failing stops being re-asked while other fields still get their turns.
REASK_MAX_ATTEMPTS_PER_FIELD = int(os.environ.get("ACCURATE_PIPE_REASK_ATTEMPTS", "2"))

# Hard ceiling on total re-ask rounds regardless of per-field counts, so a
# pathological document can't spin. Each round is one model call.
REASK_MAX_ROUNDS = int(os.environ.get("ACCURATE_PIPE_REASK_ROUNDS", "3"))

# Which model does the re-asks. Default: the same light model. Set to the
# heavier one if you want re-asks to be the escalation path.
REASK_MODEL = os.environ.get("ACCURATE_PIPE_REASK_MODEL", "") or None  # None -> use TEXT_MODEL


# ---------------------------------------------------------------------------
# 5. Verification thresholds (verify.py)
# ---------------------------------------------------------------------------

# total_ht + tva_amount + stamp_duty should equal total_ttc. Allow this
# much absolute difference for rounding (amounts are in dinars, 3 decimals
# = millimes, so 0.01 is ten millimes -- comfortably above rounding noise,
# below a real error).
TOTALS_TOLERANCE = 0.01

# Same tolerance for "line items sum to total_ht".
LINE_SUM_TOLERANCE = 0.05  # a bit looser: per-line rounding accumulates


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------

# Where run.py writes <name>.json / <name>.xml / <name>.report.txt
OUTPUT_DIR = os.environ.get("ACCURATE_PIPE_OUT", "data/output/accurate_pipe")
