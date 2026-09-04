# -*- coding: utf-8 -*-
"""Drives adaptive_pipe's review UI backend end to end with FastAPI's
TestClient, against a temp SQLite DB and the real facture.pdf -- no
Ollama needed, since we seed the queue directly via
adaptive_pipe.queue.submit_for_review() the way pipeline.run() would,
rather than running the actual model calls.
"""

import os

import pytest
from fastapi.testclient import TestClient

from accurate_pipe import assemble
from adaptive_pipe import queue as ap_queue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACTURE_PDF = os.path.join(REPO_ROOT, "data", "docs", "facture.pdf")

FIELDS = {
    "invoice_number": "260143",
    "issue_date": "27/08/2026",
    "seller_name": "ideryet services",
    "seller_tax_id": "503873QAM000",
    "buyer_name": "DR. KAMMOUN MOHAMED MONCEF",
    "buyer_tax_id": None,     # deliberately missing, like the real run
    "total_ht": "97.500",
    "tva_amount": "18.525",
    "stamp_duty": "1.000",
    "total_ttc": "117.025",
    "line_items": [
        {"code": "000", "description": "MAIN D OEUVRE", "unit_price_ht": "97.500",
         "quantity": "1", "vat_rate_percent": "19", "line_total_ht": "97.500"},
    ],
}
SOURCE_TEXT = "FACTURE N 260143\nLe : 27/08/2026\n335352FAP000\nMatricule Fiscal\nNET A PAYER 117.025\n"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from adaptive_pipe.ui import queue_bridge
    from teif_pipeline.review.store import ReviewStore

    store = ReviewStore(db_path=str(tmp_path / "test_review.db"))
    queue_bridge.set_store(store)

    from adaptive_pipe.ui import server as server_module
    return TestClient(server_module.app), store


def _seed(store):
    from accurate_pipe import verify
    problems = verify.verify(FIELDS, SOURCE_TEXT)
    invoice = assemble.to_invoice(FIELDS, source_text=SOURCE_TEXT, model_used="test")
    return ap_queue.submit_for_review(store, invoice, SOURCE_TEXT, problems,
                                      pdf_path=FACTURE_PDF, fields=FIELDS)


def test_index_page_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "adaptive_pipe review" in r.text
    assert "pdf.js" in r.text or "pdfjsLib" in r.text


def test_browse_lists_pdfs(client):
    c, _ = client
    r = c.get("/api/browse", params={"path": os.path.join(REPO_ROOT, "data", "docs")})
    assert r.status_code == 200
    names = [e["name"] for e in r.json()["entries"]]
    assert "facture.pdf" in names


def test_queue_list_and_detail(client):
    c, store = client
    item_id = _seed(store)

    listing = c.get("/api/queue").json()
    assert len(listing) == 1
    assert listing[0]["status"] == "pending"

    detail = c.get(f"/api/queue/{item_id}").json()
    assert detail["fields"]["invoice_number"] == "260143"
    assert detail["fields"]["buyer_tax_id"] is None
    assert any(p["field_name"] == "buyer_tax_id" for p in detail["problems"])


def test_correct_then_fields_reflect_it(client):
    c, store = client
    item_id = _seed(store)

    r = c.post(f"/api/queue/{item_id}/correct",
              json={"field_name": "buyer_tax_id", "corrected_value": "335352FAP000"})
    assert r.status_code == 200

    detail = c.get(f"/api/queue/{item_id}").json()
    assert detail["fields"]["buyer_tax_id"] == "335352FAP000"
    assert detail["status"] == "corrected"

    # and it's stored under the DOTTED path, so memory.py can find it later
    corr = store.corrections_for(item_id)
    assert corr[0]["field_name"] == "buyer.tax_id"


def test_highlights_locate_real_values_in_the_real_pdf(client):
    c, store = client
    item_id = _seed(store)
    r = c.get(f"/api/queue/{item_id}/highlights")
    assert r.status_code == 200
    hl = r.json()
    # total_ttc = 117.025 genuinely appears in facture.pdf's text layer
    assert "total_ttc" in hl
    assert hl["total_ttc"]["rects"][0]["precise"] is True
    assert hl["total_ttc"]["color"].startswith("#")


def test_pdf_bytes_served(client):
    c, store = client
    item_id = _seed(store)
    r = c.get(f"/api/queue/{item_id}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_xml_reflects_correction(client):
    c, store = client
    item_id = _seed(store)
    c.post(f"/api/queue/{item_id}/correct",
          json={"field_name": "buyer_tax_id", "corrected_value": "335352FAP000"})
    r = c.get(f"/api/queue/{item_id}/xml")
    assert "<TEIF" in r.json()["xml"]
    assert "335352FAP000" in r.json()["xml"]


def test_approve(client):
    c, store = client
    item_id = _seed(store)
    r = c.post(f"/api/queue/{item_id}/approve")
    assert r.status_code == 200
    assert c.get(f"/api/queue/{item_id}").json()["status"] == "approved"


def test_process_endpoint_rejects_missing_file(client):
    c, _ = client
    r = c.post("/api/process", json={"paths": ["nope.pdf"]})
    assert r.status_code == 400


def test_404_for_unknown_item(client):
    c, _ = client
    assert c.get("/api/queue/99999").status_code == 404


def test_delete_removes_item_and_its_corrections(client):
    c, store = client
    item_id = _seed(store)
    c.post(f"/api/queue/{item_id}/correct",
          json={"field_name": "buyer_tax_id", "corrected_value": "335352FAP000"})
    assert len(store.corrections_for(item_id)) == 1

    r = c.delete(f"/api/queue/{item_id}")
    assert r.status_code == 200
    assert c.get(f"/api/queue/{item_id}").status_code == 404
    assert store.corrections_for(item_id) == []
    assert c.get("/api/queue").json() == []


def test_delete_404_for_unknown_item(client):
    c, _ = client
    assert c.delete("/api/queue/99999").status_code == 404


def test_field_spec_lists_every_canonical_field(client):
    """The doc-pane always shows every field the pipeline can extract --
    this is the endpoint that makes a completely omitted field visible."""
    c, _ = client
    r = c.get("/api/field-spec")
    assert r.status_code == 200
    keys = [f["key"] for f in r.json()["fields"]]
    assert "buyer_tax_id" in keys and "stamp_duty" in keys
    li_keys = [f["key"] for f in r.json()["line_item_fields"]]
    assert "description" in li_keys and "line_total_ht" in li_keys


def test_process_rejects_when_no_paths(client):
    c, _ = client
    r = c.post("/api/process", json={"paths": []})
    assert r.status_code == 200  # empty batch is valid, just does nothing
    job = r.json()
    assert job["progress"]["total"] == 0
