# -*- coding: utf-8 -*-
"""adaptive_pipe.pipeline: the tiering/escalation orchestration, end to
end but with no Ollama -- extract.extract_once and the re-ask loop's
model call are monkeypatched with scripted fakes; ingest is monkeypatched
to skip PDF/vision handling entirely (that machinery is accurate_pipe's,
already covered by its own tests).

What this file actually verifies is the NEW behavior: escalation fires on
verify.py's result (not a model self-report), a strictly-better heavy
result is adopted, a heavy result that doesn't improve is discarded, a
still-failing document is submitted to the real review queue and gets NO
xml, and a clean primary-tier pass never even calls the escalation model.
"""

import json

import pytest

from accurate_pipe import extract as accurate_extract  # for ExtractionResult shape ref only
from accurate_pipe import llm as accurate_llm
from accurate_pipe import reask as accurate_reask
from accurate_pipe.ingest import IngestResult
from adaptive_pipe import extract as ap_extract
from adaptive_pipe import pipeline as ap_pipeline
from teif_pipeline.review.store import ReviewStore

SOURCE_TEXT = """\
ideryet services
FACTURE N 260143
Le : 27/08/2026
Nom Client DR. KAMMOUN MOHAMED MONCEF
335352FAP000
Matricule Fiscal
000 MAIN D OEUVRE 97.500 1 19 97.500
Total Hors TVA 97.500
Cumul TVA 18.525
Droit de Timbre 1.000
NET A PAYER 117.025
Code TVA : 503873QAM000
"""

GOOD_FIELDS = {
    "invoice_number": "260143",
    "issue_date": "27/08/2026",
    "seller_name": "ideryet services",
    "seller_tax_id": "503873QAM000",
    "buyer_name": "DR. KAMMOUN MOHAMED MONCEF",
    "buyer_tax_id": "335352FAP000",
    "total_ht": "97.500",
    "tva_amount": "18.525",
    "stamp_duty": "1.000",
    "total_ttc": "117.025",
    "line_items": [{"code": "000", "description": "MAIN D OEUVRE",
                    "unit_price_ht": "97.500", "quantity": "1",
                    "vat_rate_percent": "19", "line_total_ht": "97.500"}],
    "_verification": {"checks_done": [], "problems_found": []},
}


def _deficient_fields():
    f = dict(GOOD_FIELDS)
    f["buyer_tax_id"] = None
    return f


def _fake_ingest(pdf_path):
    return IngestResult(text=SOURCE_TEXT, classification="text", low_confidence_source=False)


def _fake_reask_never_fixes(messages, schema, model=None, extra_options=None):
    # returns null for whatever was asked -- the re-ask round accepts nothing
    want = {k: None for k in schema["properties"]}
    return accurate_llm.LLMResult(text=json.dumps(want), model=model or "fake",
                                  duration_s=0.0, prompt_tokens=1, output_tokens=1)


@pytest.fixture
def store(tmp_path):
    return ReviewStore(db_path=str(tmp_path / "review.db"))


def test_light_tier_clean_pass_never_calls_heavy(monkeypatch, store):
    monkeypatch.setattr(ap_pipeline.ingest, "ingest", _fake_ingest)

    def light_extract(text, model):
        assert model == ap_pipeline.config.TEXT_MODEL
        return ap_extract.ExtractionResult(fields=dict(GOOD_FIELDS), model=model)

    def heavy_should_not_run(text, model):
        raise AssertionError("escalation should not run when the primary tier is clean")

    calls = {"n": 0}

    def which_extract(text, model):
        calls["n"] += 1
        if model == ap_pipeline.config.TEXT_MODEL:
            return light_extract(text, model)
        return heavy_should_not_run(text, model)

    monkeypatch.setattr(ap_pipeline.extract, "extract_once", which_extract)
    monkeypatch.setattr(accurate_reask.llm, "chat_json", _fake_reask_never_fixes)

    result = ap_pipeline.run("fake.pdf", store=store)

    assert result.tier_used == "primary"
    assert len(result.tiers_tried) == 1
    assert not result.needs_review
    assert result.review_item_id is None
    assert result.xml is not None
    assert "<TEIF" in result.xml


def test_heavy_tier_adopted_when_it_fixes_the_error(monkeypatch, store):
    monkeypatch.setattr(ap_pipeline.ingest, "ingest", _fake_ingest)
    # escalation is off by default now (single qwen3 tier -- see config.py);
    # this test specifically exercises the escalation path, so opt in --
    # and give the two tiers DISTINCT models, since the new single-tier
    # default would otherwise make both equal to "qwen3:14b".
    monkeypatch.setattr(ap_pipeline.config, "TEXT_MODEL", "mistral:latest")
    monkeypatch.setattr(ap_pipeline.config, "TEXT_MODEL_ESCALATION", "qwen3:14b")

    def which_extract(text, model):
        if model == ap_pipeline.config.TEXT_MODEL:
            return ap_extract.ExtractionResult(fields=_deficient_fields(), model=model)
        assert model == ap_pipeline.config.TEXT_MODEL_ESCALATION
        return ap_extract.ExtractionResult(fields=dict(GOOD_FIELDS), model=model)

    monkeypatch.setattr(ap_pipeline.extract, "extract_once", which_extract)
    monkeypatch.setattr(accurate_reask.llm, "chat_json", _fake_reask_never_fixes)

    result = ap_pipeline.run("fake.pdf", store=store)

    assert result.tier_used == "escalated"
    assert len(result.tiers_tried) == 2
    assert not result.needs_review
    assert result.xml is not None
    assert result.fields["buyer_tax_id"] == "335352FAP000"


def test_both_tiers_failing_goes_to_review_with_no_xml(monkeypatch, store):
    monkeypatch.setattr(ap_pipeline.ingest, "ingest", _fake_ingest)

    def which_extract(text, model):
        return ap_extract.ExtractionResult(fields=_deficient_fields(), model=model)

    monkeypatch.setattr(ap_pipeline.extract, "extract_once", which_extract)
    monkeypatch.setattr(accurate_reask.llm, "chat_json", _fake_reask_never_fixes)

    result = ap_pipeline.run("fake.pdf", store=store)

    assert result.needs_review
    assert result.xml is None
    assert result.review_item_id is not None

    # and it's really in the store -- not a fake side channel
    item = store.get(result.review_item_id)
    assert item is not None
    assert item.status == "pending"
    assert "260143" in item.source_text


def test_on_progress_reports_each_phase_including_escalation(monkeypatch, store):
    monkeypatch.setattr(ap_pipeline.ingest, "ingest", _fake_ingest)
    monkeypatch.setattr(ap_pipeline.config, "TEXT_MODEL", "mistral:latest")
    monkeypatch.setattr(ap_pipeline.config, "TEXT_MODEL_ESCALATION", "qwen3:14b")

    def which_extract(text, model):
        if model == ap_pipeline.config.TEXT_MODEL:
            return ap_extract.ExtractionResult(fields=_deficient_fields(), model=model)
        return ap_extract.ExtractionResult(fields=dict(GOOD_FIELDS), model=model)

    monkeypatch.setattr(ap_pipeline.extract, "extract_once", which_extract)
    monkeypatch.setattr(accurate_reask.llm, "chat_json", _fake_reask_never_fixes)

    seen = []
    result = ap_pipeline.run("fake.pdf", store=store, on_progress=seen.append)

    assert not result.needs_review
    assert seen[0].startswith("Reading document text")
    # phase text names the actual model, never a light/heavy label
    assert any("Extracting fields with mistral:latest" in p for p in seen)
    assert any("Escalating to qwen3:14b" in p for p in seen)
    assert any("Extracting fields with qwen3:14b" in p for p in seen)
    assert seen[-1] == "Done."


def test_heavy_result_not_adopted_when_it_does_not_improve(monkeypatch, store):
    monkeypatch.setattr(ap_pipeline.ingest, "ingest", _fake_ingest)
    monkeypatch.setattr(ap_pipeline.config, "TEXT_MODEL_ESCALATION", "qwen3:14b")

    def which_extract(text, model):
        # both tiers equally broken
        return ap_extract.ExtractionResult(fields=_deficient_fields(), model=model)

    monkeypatch.setattr(ap_pipeline.extract, "extract_once", which_extract)
    monkeypatch.setattr(accurate_reask.llm, "chat_json", _fake_reask_never_fixes)

    result = ap_pipeline.run("fake.pdf", store=store)

    # both tiers were tried, but the report should say heavy didn't help
    assert len(result.tiers_tried) == 2
    assert any("did not improve" in n or "kept the primary tier" in n for n in result.notes)
