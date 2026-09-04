# -*- coding: utf-8 -*-
"""adaptive_pipe.memory: the per-seller memory built from real
teif_pipeline.review corrections. Uses a real (temp-file) ReviewStore --
this is deliberately not mocked, since the whole point of this module is
correctly reading back what that store actually persists.

Covers the two paths:
  PATH 1 apply_direct_reuse  -- only fills a stable field if the historical
         value is ALSO found verbatim in the CURRENT document.
  PATH 2 hints_from_history  -- builds a positional hint from where a past
         corrected value sat relative to a label, never supplies the value.
"""

import json

import pytest

from adaptive_pipe import memory
from teif_pipeline.review.store import ReviewStore

SELLER_TAX_ID = "503873QAM000"


@pytest.fixture
def store(tmp_path):
    return ReviewStore(db_path=str(tmp_path / "review.db"))


def _submit(store, source_text, seller_tax_id=SELLER_TAX_ID, buyer_name=None):
    invoice_json = json.dumps({
        "seller": {"tax_id": seller_tax_id, "name": None},
        "buyer": {"name": buyer_name},
    })
    return store.submit(source_text=source_text, invoice_json=invoice_json,
                        gate_results_json="[]", source_path="x.pdf")


# ---------------------------------------------------------------------------
# PATH 1: direct reuse
# ---------------------------------------------------------------------------

def test_direct_reuse_only_when_value_reappears(store):
    old_text = "ideryet services\nSARL au capital de 450 000 dt\nFACTURE N 1"
    item1 = _submit(store, old_text)
    store.correct_field(item1, "seller.name", None, "ideryet services")

    # new document, same seller, name missing again, value IS present here too
    new_text_ok = "ideryet services\nEnterprise Solutions\nFACTURE N 2"
    fields, notes = memory.apply_direct_reuse(
        store, SELLER_TAX_ID, {"seller_tax_id": SELLER_TAX_ID, "seller_name": None}, new_text_ok
    )
    assert fields["seller_name"] == "ideryet services"
    assert "reused" in notes[0]

    # a different document where the old value does NOT appear -> must not reuse
    new_text_bad = "a completely different invoice with no seller name printed"
    fields2, notes2 = memory.apply_direct_reuse(
        store, SELLER_TAX_ID, {"seller_tax_id": SELLER_TAX_ID, "seller_name": None}, new_text_bad
    )
    assert fields2["seller_name"] is None
    assert "does NOT appear" in notes2[0]


def test_direct_reuse_never_overrides_an_already_present_value(store):
    old_text = "ACME Corp letterhead"
    item1 = _submit(store, old_text)
    store.correct_field(item1, "seller.name", None, "ACME Corp")

    fields, notes = memory.apply_direct_reuse(
        store, SELLER_TAX_ID,
        {"seller_tax_id": SELLER_TAX_ID, "seller_name": "Something Else Extracted"},
        "ACME Corp appears here too",
    )
    assert fields["seller_name"] == "Something Else Extracted"   # untouched
    assert notes == []


def test_direct_reuse_ignores_other_sellers(store):
    old_text = "Other Seller Ltd letterhead"
    item1 = _submit(store, old_text, seller_tax_id="999999ZZZ999")
    store.correct_field(item1, "seller.name", None, "Other Seller Ltd")

    fields, notes = memory.apply_direct_reuse(
        store, SELLER_TAX_ID, {"seller_tax_id": SELLER_TAX_ID, "seller_name": None},
        "Other Seller Ltd",  # even though the string happens to appear
    )
    assert fields["seller_name"] is None   # no history for THIS seller
    assert notes == []


def test_direct_reuse_no_op_without_seller_tax_id(store):
    fields, notes = memory.apply_direct_reuse(store, None, {"seller_name": None}, "some text")
    assert fields == {"seller_name": None}
    assert notes == []


# ---------------------------------------------------------------------------
# PATH 2: positional hints
# ---------------------------------------------------------------------------

def test_hint_extracts_the_real_facture_pattern(store):
    # the actual scrambled-body pattern from facture.pdf's source text
    old_text = (
        "C.C.P. : 17 701 0000 000 594843 96 Ville SFAX\n"
        "335352FAP000\n"
        "Le : 27/08/2026 Matricule Fiscal\n"
    )
    item1 = _submit(store, old_text)
    store.correct_field(item1, "buyer.tax_id", None, "335352FAP000")

    hints = memory.hints_from_history(store, SELLER_TAX_ID, ["buyer_tax_id"])
    assert "buyer_tax_id" in hints
    assert "Matricule Fiscal" in hints["buyer_tax_id"]
    assert "on a prior invoice" in hints["buyer_tax_id"]   # only 1 example so far


def test_hint_confidence_wording_scales_with_repetition(store):
    text = "335352FAP000\nMatricule Fiscal\n"
    for _ in range(2):
        item = _submit(store, text)
        store.correct_field(item, "buyer.tax_id", None, "335352FAP000")

    hints = memory.hints_from_history(store, SELLER_TAX_ID, ["buyer_tax_id"])
    assert "consistently" in hints["buyer_tax_id"]   # >= MIN_CORRECTIONS_FOR_CONFIDENT_HINT


def test_hint_absent_when_no_history_for_field(store):
    hints = memory.hints_from_history(store, SELLER_TAX_ID, ["buyer_tax_id", "issue_date"])
    assert hints == {}


def test_hint_never_returns_the_value_itself(store):
    """The hint must describe a POSITION, never leak the historical value
    as if it were the answer -- that would defeat the whole point."""
    item1 = _submit(store, "335352FAP000\nMatricule Fiscal\n")
    store.correct_field(item1, "buyer.tax_id", None, "335352FAP000")

    hints = memory.hints_from_history(store, SELLER_TAX_ID, ["buyer_tax_id"])
    assert "335352FAP000" not in hints["buyer_tax_id"]
