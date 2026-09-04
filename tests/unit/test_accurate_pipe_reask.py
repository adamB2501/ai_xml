# -*- coding: utf-8 -*-
"""accurate_pipe re-ask loop: does it (a) fix a recoverable field, and
(b) give up after the per-field cap and flag the field for review.

No Ollama - `llm.chat_json` is monkeypatched with a scripted fake model.
"""

import json

from accurate_pipe import config, llm, reask, verify

# every value in BASE_FIELDS that we keep DOES appear here, so the
# occurrence check is quiet unless a test deliberately breaks a field.
SRC = """\
ideryet services
SARL au capital de 450 000 dt
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

BASE_FIELDS = {
    "invoice_number": "260143",
    "issue_date": "27/08/2026",
    "seller_name": "ideryet services",
    "seller_tax_id": "503873QAM000",
    "buyer_name": "DR. KAMMOUN MOHAMED MONCEF",
    "buyer_tax_id": None,          # missing - recoverable from SRC
    "total_ht": "97.500",
    "tva_amount": "18.525",
    "total_ttc": "117.025",
    "stamp_duty": None,            # missing - recoverable, totals imply 1.000
    "line_items": [{"code": "000", "description": "MAIN D OEUVRE",
                    "line_total_ht": "97.500"}],
    "_verification": {"checks_done": [], "problems_found": []},
}


def _fake(answers_by_round):
    state = {"i": 0}

    def chat_json(messages, schema, model=None, extra_options=None):
        want = set(schema["properties"])
        src = answers_by_round[min(state["i"], len(answers_by_round) - 1)]
        state["i"] += 1
        ans = {k: v for k, v in src.items() if k in want}
        return llm.LLMResult(text=json.dumps(ans), model="fake",
                             duration_s=0.0, prompt_tokens=1, output_tokens=1)

    return chat_json


def test_reask_recovers_missing_fields(monkeypatch):
    monkeypatch.setattr(
        reask.llm, "chat_json",
        _fake([{"buyer_tax_id": "335352FAP000", "stamp_duty": "1.000"}]),
    )
    problems = verify.verify(BASE_FIELDS, SRC)
    assert verify.needs_human_review(problems)          # broken before

    r = reask.run(dict(BASE_FIELDS), SRC, problems)

    assert r.fields["buyer_tax_id"] == "335352FAP000"
    assert r.fields["stamp_duty"] == "1.000"
    assert r.still_deficient == []
    assert not verify.needs_human_review(r.problems)
    assert len(r.llm_calls) == 1


def test_reask_rejects_value_not_in_source(monkeypatch):
    monkeypatch.setattr(
        reask.llm, "chat_json",
        _fake([{"buyer_tax_id": "111111AAA111", "stamp_duty": "1.000"}]),
    )
    r = reask.run(dict(BASE_FIELDS), SRC, verify.verify(BASE_FIELDS, SRC))

    assert r.fields["buyer_tax_id"] is None             # never accepted
    assert r.fields["stamp_duty"] == "1.000"            # good one still landed
    assert "buyer_tax_id" in r.still_deficient
    assert r.attempts["buyer_tax_id"] == config.REASK_MAX_ATTEMPTS_PER_FIELD


def test_max_rounds_override_caps_tighter_than_the_shared_default(monkeypatch):
    """A caller (adaptive_pipe, running a slower model) can cap the loop
    below config.REASK_MAX_ROUNDS without changing that shared default for
    every other caller."""
    calls = {"n": 0}

    def counting_fake(messages, schema, model=None, extra_options=None):
        calls["n"] += 1
        # never actually resolves anything -> the loop would keep going
        # every round up to whatever cap is in effect
        return llm.LLMResult(text=json.dumps({}), model="fake",
                             duration_s=0.0, prompt_tokens=1, output_tokens=1)

    monkeypatch.setattr(reask.llm, "chat_json", counting_fake)
    assert config.REASK_MAX_ROUNDS > 2   # precondition: the override is actually tighter

    r = reask.run(dict(BASE_FIELDS), SRC, verify.verify(BASE_FIELDS, SRC), max_rounds=2)

    assert calls["n"] == 2               # stopped at the override, not the shared default
    assert max(r.attempts.values()) <= 2


def test_reask_not_triggered_when_first_pass_is_clean(monkeypatch):
    good = dict(BASE_FIELDS)
    good["buyer_tax_id"] = "335352FAP000"
    good["stamp_duty"] = "1.000"

    def boom(*a, **k):
        raise AssertionError("re-ask should not have called the model")

    monkeypatch.setattr(reask.llm, "chat_json", boom)

    problems = verify.verify(good, SRC)
    assert not verify.needs_human_review(problems)
    r = reask.run(good, SRC, problems)
    assert r.rounds == []
    assert r.still_deficient == []


def test_warning_only_field_does_not_block(monkeypatch):
    # seller_capital carried a benign occurrence warning and the model
    # can't do better -> re-asked, left as a warning, but NOT blocking.
    fields = dict(BASE_FIELDS)
    fields["buyer_tax_id"] = "335352FAP000"
    fields["stamp_duty"] = "1.000"
    fields["seller_capital"] = "450000"          # printed as "450 000" -> occurrence warn

    monkeypatch.setattr(reask.llm, "chat_json", _fake([{"seller_capital": "450000"}]))
    r = reask.run(fields, SRC, verify.verify(fields, SRC))

    assert "seller_capital" not in r.still_deficient   # warning != blocking
    assert not verify.needs_human_review(r.problems)


def test_hint_computes_implied_stamp_duty():
    hints = reask.build_hints(BASE_FIELDS)
    assert "stamp_duty" in hints and "1" in hints["stamp_duty"]
    assert "buyer_tax_id" in hints and "503873QAM000" in hints["buyer_tax_id"]
