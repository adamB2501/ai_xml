# -*- coding: utf-8 -*-
"""
The whole pipeline. Read this to see the flow; detail is in the modules.

    PDF
     |
     v
  accurate_pipe.ingest.ingest(pdf)         ->  source text
     |                                          (pdfplumber + vision model, unchanged)
     v
  extract.extract_once(text, PRIMARY model) ->  fields dict
     |
     v
  memory.apply_direct_reuse(...)           ->  fields dict  (PATH 1: only fills a
     |                                          field if a past-confirmed value for
     |                                          THIS seller is found VERBATIM in
     |                                          THIS document -- never injected blind)
     v
  accurate_pipe.verify.verify(...)         ->  list[Problem]
     |
     v
  accurate_pipe.reask.run(..., model=PRIMARY,
      extra_hints=memory.hints_from_history,
      max_rounds=config.REASK_MAX_ROUNDS)   ->  fields dict + list[Problem]
     |                                          (PATH 2: sharper, seller-specific
     |                                           hints on top of the generic ones;
     |                                           still has to find + verify the
     |                                           value on THIS document. Capped at
     |                                           REASK_MAX_ROUNDS calls, same model.)
     v
  still failing?  --no-->  assemble + build TEIF XML + validate  -->  done, no human
     |
     yes, AND TEXT_MODEL_ESCALATION is set (off by default -- see config.py)
     v
  repeat extract -> memory -> verify -> reask, with the ESCALATION model
     |
     v
  still failing?  --no-->  assemble + build TEIF XML + validate  -->  done, no human
     |
     yes
     v
  queue.submit_for_review(...)  ->  teif_pipeline.review's real SQLite queue.
  NO filing-ready XML is produced. A human corrects fields there; those
  corrections become tomorrow's memory hints for this same seller.

Naming note: tiers are called "primary" and "escalated", not "light" and
"heavy" -- with TEXT_MODEL defaulting to qwen3:14b (see config.py, changed
after the lighter mistral-first design added latency without resolving
more documents on these real invoices), "light" would describe a tier
that isn't light at all. The phase text shown to the UI names the actual
model, never a weight-implying label.

The escalation trigger is the actual fix over accurate_pipe: it fires on
verify.py's deterministic result, not on a model's own self-report (which
we measured under-reporting real problems -- see adaptive_pipe/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from accurate_pipe import assemble, ingest, verify
from accurate_pipe import reask as reask_mod
from teif_pipeline.teif.builder import build_teif_xml
from teif_pipeline.teif.validate import Finding, validate_teif_xml

from . import config, extract, memory, queue

# a no-op default so every call site can just do on_progress(phase) without
# an `if on_progress:` guard everywhere
_NOOP = lambda phase: None  # noqa: E731


@dataclass
class TierResult:
    tier: str                 # "primary" | "escalated"
    fields: dict
    problems: list
    still_deficient: list
    memory_notes: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for p in self.problems if p.severity == "error")


@dataclass
class AdaptiveResult:
    pdf_path: str
    source_text: str
    classification: str
    low_confidence_source: bool

    fields: dict
    tier_used: str             # "primary" | "escalated"
    tiers_tried: list[TierResult]

    problems: list
    needs_review: bool
    review_item_id: int | None

    invoice: object                          # teif_pipeline.models.Invoice
    xml: str | None                          # None when needs_review -- see module docstring
    build_warnings: list[str]
    xml_findings: list[Finding]

    notes: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)


def _run_tier(tier_name: str, model: str, source_text: str, seller_tax_id_hint: str | None,
             store, on_progress: Callable[[str], None]) -> TierResult:
    # phase text always names the actual model, never a "light"/"heavy"
    # label -- see the module docstring's naming note.
    on_progress(f"Extracting fields with {model}…")
    ext = extract.extract_once(source_text, model=model)
    fields = ext.fields
    calls = list(ext.llm_calls)

    seller_tax_id = fields.get("seller_tax_id") or seller_tax_id_hint
    mem_notes: list[str] = []
    if seller_tax_id:
        fields, mem_notes = memory.apply_direct_reuse(store, seller_tax_id, fields, source_text)

    on_progress(f"Checking extracted fields against the document…")
    problems = verify.verify(fields, source_text)

    deficient = reask_mod.deficient_fields(fields, problems)
    extra_hints = memory.hints_from_history(store, seller_tax_id, deficient) if seller_tax_id else {}

    def _on_round(round_no: int, wanted: list[str]) -> None:
        shown = ", ".join(wanted[:3]) + ("…" if len(wanted) > 3 else "")
        on_progress(f"Refining fields with {model} (round {round_no}/{config.REASK_MAX_ROUNDS}): {shown}")

    rk = reask_mod.run(fields, source_text, problems, model=model,
                       extra_hints=extra_hints, on_round=_on_round,
                       max_rounds=config.REASK_MAX_ROUNDS)
    calls += rk.llm_calls

    return TierResult(
        tier=tier_name, fields=rk.fields, problems=rk.problems,
        still_deficient=rk.still_deficient, memory_notes=mem_notes, llm_calls=calls,
    )


def run(pdf_path: str, store=None, on_progress: Optional[Callable[[str], None]] = None) -> AdaptiveResult:
    """`on_progress(phase_text)` is called at each major step, purely for
    UI reporting (see adaptive_pipe/ui/jobs.py) -- it never affects what
    the pipeline does. Optional, defaults to a no-op."""
    on_progress = on_progress or _NOOP
    store = store or queue.open_store()
    notes: list[str] = []
    all_calls: list = []
    tiers: list[TierResult] = []

    # -- STEP 1: PDF -> source text (unchanged accurate_pipe machinery) --
    on_progress("Reading document text (may include a slow vision-model pass "
               "for a scanned page or a rasterized letterhead/footer)…")
    ing = ingest.ingest(pdf_path)
    notes += [f"[ingest] {n}" for n in ing.notes]
    all_calls += ing.vision_calls

    # -- STEP 2: primary tier -------------------------------------------
    primary = _run_tier("primary", config.TEXT_MODEL, ing.text, None, store, on_progress)
    tiers.append(primary)
    all_calls += primary.llm_calls
    notes += [f"[primary/memory] {n}" for n in primary.memory_notes]
    notes.append(f"[primary] {primary.error_count} error(s), "
                 f"still_deficient={primary.still_deficient}")

    best = primary

    # -- STEP 3: escalate only if the primary tier is actually still broken -
    primary_failing = verify.needs_human_review(primary.problems) or bool(primary.still_deficient)
    if primary_failing and config.TEXT_MODEL_ESCALATION:
        on_progress(f"Escalating to {config.TEXT_MODEL_ESCALATION}…")
        notes.append(
            f"[escalate] verify.py found real problems after the primary tier's "
            f"own re-ask loop -> trying {config.TEXT_MODEL_ESCALATION} fresh "
            f"(not the model's self-report -- that's the fix over accurate_pipe)"
        )
        escalated = _run_tier("escalated", config.TEXT_MODEL_ESCALATION, ing.text,
                              primary.fields.get("seller_tax_id"), store, on_progress)
        tiers.append(escalated)
        all_calls += escalated.llm_calls
        notes += [f"[escalated/memory] {n}" for n in escalated.memory_notes]
        notes.append(f"[escalated] {escalated.error_count} error(s), "
                     f"still_deficient={escalated.still_deficient}")

        # prefer the escalated result only if it's actually better -- fewer
        # errors, or same errors but fewer unresolved fields. Never regress
        # silently.
        if (escalated.error_count, len(escalated.still_deficient)) < (best.error_count, len(best.still_deficient)):
            best = escalated
            notes.append("[escalate] kept the escalated tier's result (strictly better)")
        else:
            notes.append("[escalate] kept the primary tier's result (escalation didn't improve on it)")

    fields = best.fields
    problems = best.problems

    # -- STEP 4: assemble the Invoice either way (needed even for review) -
    invoice = assemble.to_invoice(
        fields, source_text=ing.text,
        model_used=f"adaptive_pipe:{best.tier}", notes=list(notes),
    )

    needs_review = (
        verify.needs_human_review(problems)
        or bool(best.still_deficient)
        or ing.low_confidence_source
    )
    if ing.low_confidence_source:
        notes.append("[verdict] source is a scan / OCR-layer page -> review required "
                     "regardless of checks.")

    review_item_id = None
    xml, build_warnings, xml_findings = None, [], []

    if needs_review:
        on_progress("Submitting for human review…")
        review_item_id = queue.submit_for_review(store, invoice, ing.text, problems, pdf_path, fields=fields)
        notes.append(f"[queue] submitted as review_items.id={review_item_id} -- "
                     f"NOT building a filing-ready XML until a human resolves this.")
    else:
        on_progress("Finalizing — building TEIF XML…")
        xml, build_warnings = build_teif_xml(invoice)
        xml_findings = validate_teif_xml(xml)
        if any(getattr(f, "severity", "") == "error" for f in xml_findings):
            # the field-level checks passed but XML-level validation still
            # caught something -- don't file it either, queue it instead.
            needs_review = True
            on_progress("XML validation failed — submitting for human review…")
            review_item_id = queue.submit_for_review(store, invoice, ing.text, problems, pdf_path, fields=fields)
            notes.append(f"[queue] XML validation still found an error -> "
                         f"submitted as review_items.id={review_item_id}")
            xml = None

    on_progress("Done.")
    return AdaptiveResult(
        pdf_path=pdf_path,
        source_text=ing.text,
        classification=ing.classification,
        low_confidence_source=ing.low_confidence_source,
        fields=fields,
        tier_used=best.tier,
        tiers_tried=tiers,
        problems=problems,
        needs_review=needs_review,
        review_item_id=review_item_id,
        invoice=invoice,
        xml=xml,
        build_warnings=build_warnings,
        xml_findings=xml_findings,
        notes=notes,
        llm_calls=all_calls,
    )
