# -*- coding: utf-8 -*-
"""
The per-seller memory layer, built from teif_pipeline.review's real
correction log (SQLite: review_items + corrections -- see that package's
store.py, unchanged, reused as-is).

THE RULE THIS WHOLE FILE OBEYS: a past correction is a candidate to test,
never a value to trust. Nothing in here writes a field directly into the
document's fields dict. Every function returns either (a) a value that has
ALREADY been checked against THIS document's own text, or (b) a hint
string for the model to go check itself. Trusting a cached value without
re-checking it is exactly the mistake this design started from ("we're
not 100% sure these are constants") -- see the README for the walkthrough
that motivated this file.

Two paths, matching the two kinds of fields:

  PATH 1 - apply_direct_reuse()
    For config.STABLE_SELLER_FIELDS (the letterhead block: name, address,
    RC number, capital). A seller's own identity doesn't change invoice to
    invoice, so if a past-corrected value for one of these fields is
    found VERBATIM in the current document's text, it's used -- not
    because it was seen before, but because it's confirmed again, right
    now, on this document. No model call spent on it either way.

  PATH 2 - hints_from_history()
    For everything else (buyer tax id, amounts, dates, ...). These
    legitimately differ per invoice, so a past value is useless as a
    candidate. What IS useful: where, relative to what label, that kind
    of value tended to sit on this seller's past invoices. That becomes
    one more hint merged into reask.py's prompt -- the model still has to
    find and the pipeline still has to verify the actual value on THIS
    document.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from teif_pipeline.review.store import ReviewStore

from . import config


# ---------------------------------------------------------------------------
# reading the log
# ---------------------------------------------------------------------------

def _seller_tax_id_of(store: ReviewStore, review_item_id: int) -> str | None:
    item = store.get(review_item_id)
    if item is None:
        return None
    try:
        return json.loads(item.invoice_json).get("seller", {}).get("tax_id")
    except (json.JSONDecodeError, AttributeError):
        return None


def corrections_for_seller(store: ReviewStore, seller_tax_id: str) -> list[dict]:
    """Every correction on file whose document came from this seller
    (matched by seller_tax_id, the one field that both identifies a
    seller and is itself always freshly re-extracted, never assumed).
    Oldest first, so "most recent" is simply the last entry."""
    if not seller_tax_id:
        return []
    out = [
        c for c in store.all_corrections()
        if _seller_tax_id_of(store, c["review_item_id"]) == seller_tax_id
    ]
    out.sort(key=lambda c: c["corrected_at"])
    return out


def _by_dotted_field(corrections: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in corrections:
        grouped[c["field_name"]].append(c)
    return grouped


# ---------------------------------------------------------------------------
# PATH 1: direct reuse for stable seller-side fields
# ---------------------------------------------------------------------------

def apply_direct_reuse(
    store: ReviewStore, seller_tax_id: str, fields: dict, source_text: str,
) -> tuple[dict, list[str]]:
    """Fills in any of config.STABLE_SELLER_FIELDS that are currently
    empty, but ONLY with a value that (a) was human-confirmed before for
    this seller AND (b) is verbatim present in THIS document's text.
    Returns (possibly-updated fields copy, human-readable notes)."""
    if not seller_tax_id:
        return fields, []

    fields = dict(fields)
    notes: list[str] = []
    grouped = _by_dotted_field(corrections_for_seller(store, seller_tax_id))

    for flat_key in config.STABLE_SELLER_FIELDS:
        if fields.get(flat_key):
            continue  # this round already has something -- don't override it
        dotted = config.FLAT_TO_DOTTED.get(flat_key)
        history = grouped.get(dotted, [])
        if not history:
            continue
        candidate = history[-1]["corrected_value"]  # most recent confirmed value
        if candidate and candidate in source_text:
            fields[flat_key] = candidate
            notes.append(
                f"{flat_key}: reused '{candidate}' confirmed for this seller "
                f"before, AND found again in this document's own text"
            )
        else:
            notes.append(
                f"{flat_key}: a prior confirmed value exists ('{candidate}') "
                f"but it does NOT appear in this document -- not reused; "
                f"seller may have changed it, or this is a different match"
            )
    return fields, notes


# ---------------------------------------------------------------------------
# PATH 2: positional hints for fields that vary per invoice
# ---------------------------------------------------------------------------

# a "label" looks like a short run of words, no digits -- distinguishes
# "Matricule Fiscal" (a label) from "17 701 0000 000 594843 96" (data)
_LABELY = re.compile(r"^[A-Za-zÀ-ÿ .:'/-]{2,40}$")


def _line_context(text: str, value: str) -> tuple[str, str] | None:
    """(text just before `value` on its line, text just after it on its
    line) -- scoped to the same line, with a one-line look-around if the
    value sits alone on its own line. NOT a fixed character window: a
    window can straddle an unrelated line and make the hint worse, not
    better (see adaptive_pipe/README.md for the real example that showed
    this)."""
    idx = text.find(value)
    if idx == -1:
        return None
    line_start = text.rfind("\n", 0, idx) + 1
    end_search = text.find("\n", idx + len(value))
    line_end = end_search if end_search != -1 else len(text)

    before = text[line_start:idx].strip(" :|\t")
    after = text[idx + len(value):line_end].strip(" :|\t")

    if not before and line_start > 0:
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        before = text[prev_start:max(prev_start, line_start - 1)].strip()
    if not after:
        next_end = text.find("\n", line_end + 1)
        after = text[line_end + 1: next_end if next_end != -1 else len(text)].strip()
    return before, after


def _labely_suffix(snippet: str, max_words: int = 4) -> str | None:
    """The longest trailing run of up to `max_words` words in `snippet`
    that reads as a clean label (no digits) -- tried longest-first so
    "Le : 27/08/2026 Matricule Fiscal" correctly yields "Matricule Fiscal"
    instead of either the whole noisy line or nothing at all."""
    words = snippet.split()
    for n in range(min(max_words, len(words)), 0, -1):
        tail = " ".join(words[-n:])
        if _LABELY.match(tail) and not any(ch.isdigit() for ch in tail):
            return tail
    return None


def _pick_label(before: str, after: str) -> str | None:
    """Prefer whichever neighbor actually reads as a label (short,
    alphabetic, no digits) over one that reads as more data. Checks the
    text right after the value first -- on the one real example this was
    built against ("335352FAP000" followed by "Matricule Fiscal"), the
    meaningful label was on that side, not before it -- but falls back to
    "before" if "after" doesn't yield one, and to a raw snippet if neither
    does (better than nothing, worse than a clean match)."""
    for snippet in (after, before):
        label = _labely_suffix(snippet) if snippet else None
        if label:
            return label
    return (after or before or None)


def hints_from_history(
    store: ReviewStore, seller_tax_id: str, deficient_fields: list[str],
) -> dict[str, str]:
    """For fields in `deficient_fields` that have prior corrections for
    this seller, build one hint sentence each from where those past
    values sat relative to a label. Fields with no history at all are
    simply absent from the returned dict (reask.py's generic hints still
    apply to them, if any)."""
    if not seller_tax_id:
        return {}

    grouped = _by_dotted_field(corrections_for_seller(store, seller_tax_id))
    hints: dict[str, str] = {}

    for flat_key in deficient_fields:
        dotted = config.FLAT_TO_DOTTED.get(flat_key)
        history = grouped.get(dotted, [])
        if not history:
            continue

        labels = []
        for c in history:
            item = store.get(c["review_item_id"])
            if item is None:
                continue
            ctx = _line_context(item.source_text, c["corrected_value"])
            if ctx:
                label = _pick_label(*ctx)
                if label:
                    labels.append(label)
        if not labels:
            continue

        common_label, n = Counter(labels).most_common(1)[0]
        confidence = (
            "consistently" if n >= config.MIN_CORRECTIONS_FOR_CONFIDENT_HINT
            else "on a prior invoice"
        )
        hints[flat_key] = (
            f"On {n} previous invoice(s) from this same seller, this field "
            f"was {confidence} found right next to the text {common_label!r}. "
            f"Look for a value in that same position on THIS document -- but "
            f"only use it if you can actually find it here."
        )
    return hints
