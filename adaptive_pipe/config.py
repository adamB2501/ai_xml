# -*- coding: utf-8 -*-
"""
Config specific to adaptive_pipe. Everything ingest/model/DPI-related is
inherited from accurate_pipe.config (imported below, not copied) -- this
file only adds what's new: where the review DB lives, which fields are
treated as candidates for direct reuse, and how much history is needed
before a hint is phrased as confident rather than tentative.
"""

import os

from accurate_pipe import config as base  # noqa: F401  (re-exported below)

# re-export the accurate_pipe settings this pipeline also needs, so callers
# only ever import adaptive_pipe.config, never reach into accurate_pipe
# directly for a shared value.
OLLAMA_URL = base.OLLAMA_URL
OLLAMA_TIMEOUT_S = base.OLLAMA_TIMEOUT_S
VISION_MODEL = base.VISION_MODEL
TEMPERATURE = base.TEMPERATURE
NUM_CTX = base.NUM_CTX
RENDER_DPI = base.RENDER_DPI
MAX_IMAGE_PX = base.MAX_IMAGE_PX

# TEXT_MODEL / TEXT_MODEL_ESCALATION are adaptive_pipe's OWN settings, not
# re-exported from accurate_pipe.config -- deliberately decoupled so this
# doesn't also change accurate_pipe's own standalone CLI/tests.
#
# Default is a SINGLE tier on qwen3:14b, no escalation. The light-then-
# heavy design assumed the light model (mistral) would resolve enough
# documents on its own to be worth the extra ~2-5 minutes it costs before
# falling through. On the real invoices this got tried on, that assumption
# didn't hold -- mistral kept failing verification and escalating anyway,
# so every document paid for both the light pass AND the heavy pass, for
# no benefit over just running the heavy pass alone. If a document set
# ever again mostly resolves on a lighter model, put "mistral:latest" back
# as TEXT_MODEL and a stronger model as TEXT_MODEL_ESCALATION to restore
# the two-tier behavior -- the pipeline logic itself didn't change, only
# this default.
TEXT_MODEL = os.environ.get("ACCURATE_PIPE_TEXT_MODEL", "qwen3:14b")
TEXT_MODEL_ESCALATION = os.environ.get("ACCURATE_PIPE_TEXT_MODEL_ESCALATION", "") or None

# Also adaptive_pipe's own setting, not accurate_pipe's shared
# REASK_MAX_ROUNDS default (3) -- capped tighter here because with
# TEXT_MODEL now a slow model by default, 3 rounds can mean 3 more
# multi-minute calls on top of the initial extraction. pipeline.py passes
# this explicitly into accurate_pipe.reask.run(max_rounds=...), so
# accurate_pipe's own standalone default is untouched.
REASK_MAX_ROUNDS = int(os.environ.get("ADAPTIVE_PIPE_REASK_ROUNDS", "2"))

# ---------------------------------------------------------------------------
# review queue (reuses teif_pipeline.review's real SQLite store)
# ---------------------------------------------------------------------------
REVIEW_DB_PATH = os.environ.get("ADAPTIVE_PIPE_REVIEW_DB", "data/output/adaptive_pipe/review_queue.db")

# ---------------------------------------------------------------------------
# per-seller memory (memory.py)
# ---------------------------------------------------------------------------

# Flat field keys (the same names extract.py / prompts.FIELD_SPEC use) that
# are genuinely constant for a given seller -- the letterhead block, not
# anything about the buyer or this specific transaction. ONLY these are
# eligible for PATH 1 (direct reuse, no model call). Deliberately excludes
# seller_tax_id itself: that field is how a seller is identified in the
# first place, so reusing it from "history keyed by itself" would be
# circular -- it must always come from this document's own extraction.
STABLE_SELLER_FIELDS = ["seller_name", "seller_address", "seller_rc_number", "seller_capital"]

# Explicit, unambiguous mapping between the flat field keys used in the
# fields dict (extract.py / verify.py / reask.py) and the dotted paths
# teif_pipeline.review.store.correct_field() expects (matching
# teif_pipeline.models.Invoice's own attribute paths). Kept as one table,
# not a guessed transformation, because getting this silently wrong would
# mean a correction never gets found again.
FLAT_TO_DOTTED = {
    "invoice_number": "invoice_number",
    "issue_date": "issue_date",
    "seller_name": "seller.name",
    "seller_tax_id": "seller.tax_id",
    "seller_address": "seller.address.raw_text",
    "seller_rc_number": "seller.rc_number",
    "seller_capital": "capital_social",
    "buyer_name": "buyer.name",
    "buyer_tax_id": "buyer.tax_id",
    "buyer_address": "buyer.address.raw_text",
    "total_ht": "total_ht",
    "tva_amount": "tva_amount",
    "stamp_duty": "stamp_duty",
    "total_ttc": "total_ttc",
}
DOTTED_TO_FLAT = {v: k for k, v in FLAT_TO_DOTTED.items()}

# A hint built from just 1 prior example is phrased tentatively ("on a
# prior invoice..."); at this many or more consistent examples it's
# phrased as an established pattern ("consistently found..."). Doesn't
# change what the hint DOES (it's still just a hint, still still checked),
# only how it's worded -- kept low on purpose since a single real seller
# rarely has more than a handful of reviewed invoices early on.
MIN_CORRECTIONS_FOR_CONFIDENT_HINT = int(
    os.environ.get("ADAPTIVE_PIPE_MIN_CONFIDENT_HINT", "2")
)
