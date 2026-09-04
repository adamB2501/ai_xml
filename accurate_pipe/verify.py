# -*- coding: utf-8 -*-
"""
STEP 3 of the pipeline:  deterministic cross-checks on the fields dict.

The model already rechecked its own answer (prompts.py PASS 2), but a model
grading its own homework is not enough for a tax document. These checks are
plain Python - no model, fully repeatable - and they are what actually
gates the result.

Each check returns zero or more Problem records. A Problem has a severity:

    "error"    - the XML would be wrong / rejected. Must go to human review.
    "warning"  - suspicious but not provably wrong. Surface it, don't block.

Nothing here MODIFIES the fields - it only reports. Correcting values is a
human's job (or a future targeted re-ask), not a silent overwrite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .numparse import to_decimal   # see numparse.py


@dataclass
class Problem:
    severity: str      # "error" | "warning"
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


# Tunisian Matricule Fiscal shape, per the TEIF guide: 6-7 digits, a letter,
# a TVA-code letter [APBFN], a category letter [MPCNE], 3 digits. We check a
# looser version (7 digits / 3 letters / 3 digits) since extracted values
# vary in spacing and the exact code letters aren't worth rejecting on.
_TAX_ID_RE = re.compile(r"^\d{6,7}[A-Z]{2,3}\d{3}$")


def _compact(s: str | None) -> str:
    return re.sub(r"[\s/\-.]", "", s).upper() if s else ""


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def check_required_present(f: dict) -> list[Problem]:
    """TEIF-mandatory fields that must not be null."""
    required = {
        "invoice_number": "Bgm/DocumentIdentifier",
        "issue_date": "Dtm (issue date)",
        "seller_tax_id": "seller MessageSenderIdentifier",
        "buyer_tax_id": "buyer PartnerIdentifier",
        "total_ttc": "InvoiceMoa I-180 (total TTC)",
    }
    out = []
    for key, where in required.items():
        if not f.get(key):
            out.append(Problem("error", key, f"missing - {where} is mandatory in TEIF"))
    return out


def check_tax_ids(f: dict) -> list[Problem]:
    """ERROR, not warning, on a shape mismatch: a real Matricule Fiscal
    always has this shape, so a value that doesn't match isn't "probably
    fine" -- it's essentially certain to be the wrong field entirely (a
    client code, a bank reference, ...), not a slightly-off tax id. A real
    run on facture.pdf shipped 'buyer_tax_id: 41140840' (the Code Client)
    as a clean pass with this at "warning" -- it occurs verbatim in the
    source (so the occurrence check passed) and doesn't collide with the
    seller's id (so the only other tax-id check passed too), so nothing
    blocked it. Mandatory-identifier shape is a strong enough signal on
    its own that it belongs at the same severity as "missing entirely".
    """
    out = []
    s, b = _compact(f.get("seller_tax_id")), _compact(f.get("buyer_tax_id"))
    for label, val in (("seller_tax_id", s), ("buyer_tax_id", b)):
        if val and not _TAX_ID_RE.match(val):
            out.append(Problem("error", label,
                               f"'{val}' doesn't match a Matricule Fiscal shape "
                               "(7 digits / 3 letters / 3 digits) - almost "
                               "certainly the wrong field, not a malformed tax id"))
    if s and b and s == b:
        out.append(Problem("error", "buyer_tax_id",
                           "seller and buyer tax id are identical - one is wrong"))
    return out


def check_totals(f: dict) -> list[Problem]:
    """total_ht + tva_amount + stamp_duty  ==  total_ttc  (within tolerance)."""
    ht = to_decimal(f.get("total_ht"))
    tva = to_decimal(f.get("tva_amount"))
    stamp = to_decimal(f.get("stamp_duty")) or 0
    ttc = to_decimal(f.get("total_ttc"))
    if ht is None or tva is None or ttc is None:
        return [Problem("warning", "totals",
                        "cannot check total_ht + tva + stamp == total_ttc "
                        "(one of them is missing)")]
    diff = abs((ht + tva + stamp) - ttc)
    if diff > config.TOTALS_TOLERANCE:
        return [Problem("error", "totals",
                        f"total_ht ({ht}) + tva ({tva}) + stamp ({stamp}) = "
                        f"{ht + tva + stamp}, but total_ttc = {ttc} "
                        f"(off by {diff})")]
    return []


def check_line_sum(f: dict) -> list[Problem]:
    """Sum of line_total_ht should equal total_ht."""
    ht = to_decimal(f.get("total_ht"))
    items = f.get("line_items") or []
    line_vals = [to_decimal(it.get("line_total_ht")) for it in items]
    known = [v for v in line_vals if v is not None]
    if ht is None or not known:
        return []
    if len(known) < len(line_vals):
        return [Problem("warning", "line_items",
                        "some line_total_ht values are missing - cannot verify "
                        "the line sum against total_ht")]
    diff = abs(sum(known) - ht)
    if diff > config.LINE_SUM_TOLERANCE:
        return [Problem("warning", "line_items",
                        f"line totals sum to {sum(known)}, but total_ht = {ht} "
                        f"(off by {diff})")]
    return []


def check_occurrence(fields: dict, source_text: str) -> list[Problem]:
    """Every scalar value we kept must actually appear in the source text.
    This is the same idea as teif_pipeline's LLM extractive check, applied
    to the whole record. Whitespace-insensitive; skips values that are
    inherently derived (currency defaulting to TND, etc.)."""
    if not source_text:
        return []
    hay = re.sub(r"\s+", "", source_text).casefold()
    skip = {"currency", "_verification", "_raw_reply", "other_references", "line_items"}
    out = []
    for key, val in fields.items():
        if key in skip or not isinstance(val, str) or not val.strip():
            continue
        needle = re.sub(r"\s+", "", val).casefold()
        if needle and needle not in hay:
            out.append(Problem("warning", key,
                               f"value '{val}' not found verbatim in the source "
                               "text - possible hallucination or misread"))
    return out


def check_model_selfreport(fields: dict) -> list[Problem]:
    """Surface whatever the model itself flagged in PASS 2, as warnings."""
    problems = (fields.get("_verification") or {}).get("problems_found") or []
    return [Problem("warning", "_verification", f"model reported: {p}") for p in problems]


# ---------------------------------------------------------------------------
# run all
# ---------------------------------------------------------------------------

ALL_CHECKS_WITH_TEXT = [check_occurrence]
ALL_CHECKS = [
    check_required_present,
    check_tax_ids,
    check_totals,
    check_line_sum,
    check_model_selfreport,
]


def verify(fields: dict, source_text: str = "") -> list[Problem]:
    problems: list[Problem] = []
    for check in ALL_CHECKS:
        problems.extend(check(fields))
    for check in ALL_CHECKS_WITH_TEXT:
        problems.extend(check(fields, source_text))
    return problems


def needs_human_review(problems: list[Problem]) -> bool:
    return any(p.severity == "error" for p in problems)
