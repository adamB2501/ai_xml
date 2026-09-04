# -*- coding: utf-8 -*-
"""
The whole pipeline, wired together. Read this file to understand the flow;
the detail is in the modules it calls.

    PDF path
      |
      v
  ingest.ingest(pdf)          STEP 1  ->  source text
      |                                   (pdfplumber + vision model as needed)
      v
  extract.extract_fields(txt) STEP 2  ->  fields dict
      |                                   (text LLM + self-recheck + escalation)
      v
  verify.verify(fields, txt)  STEP 3  ->  list[Problem]
      |                                   (deterministic arithmetic / shape / occurrence)
      v
  reask.run(fields, txt, ..)  STEP 3b ->  fields dict + list[Problem]  (updated)
      |                                   (re-ask the model for only the failing
      |                                    fields, up to a per-field retry cap;
      |                                    accept a value only if it passes the
      |                                    same checks; leftovers -> review)
      v
  assemble.to_invoice(fields) STEP 4a ->  canonical Invoice
      |
      v
  teif.build_teif_xml(inv)    STEP 4b ->  TEIF XML string  (+ builder warnings)
      |
      v
  teif.validate_teif_xml(xml) STEP 4c ->  list[Finding]    (XML-level validation)
      |
      v
  PipelineResult  (everything above, plus a `needs_review` verdict)

`needs_review` is True if - AFTER the re-ask loop - there is still any
error-severity problem (verify) or finding (XML validation), or the
re-ask loop exhausted its retries on a blocking field, or the source was
low-confidence (a scan). A True verdict means: do NOT auto-file this XML;
a human checks it first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from teif_pipeline.teif.builder import build_teif_xml
from teif_pipeline.teif.validate import Finding, validate_teif_xml

from . import assemble, extract, ingest, reask, verify
from .llm import LLMResult


@dataclass
class PipelineResult:
    pdf_path: str

    # step 1
    source_text: str
    classification: str
    low_confidence_source: bool

    # step 2
    fields: dict
    text_model_used: str
    escalated: bool

    # step 3 (post re-ask)
    problems: list[verify.Problem]

    # step 3b
    reask_rounds: list[reask.RoundLog]
    reask_attempts: dict            # field -> times re-asked
    still_deficient: list[str]      # fields the loop could not fix

    # step 4
    invoice: object                 # teif_pipeline.models.Invoice
    xml: str
    build_warnings: list[str]
    xml_findings: list[Finding]

    # verdict + bookkeeping
    needs_review: bool
    notes: list[str] = field(default_factory=list)
    llm_calls: list[LLMResult] = field(default_factory=list)

    # convenience
    @property
    def errors(self) -> list[str]:
        out = [str(p) for p in self.problems if p.severity == "error"]
        out += [repr(f) for f in self.xml_findings if getattr(f, "severity", "") == "error"]
        return out


def run(pdf_path: str) -> PipelineResult:
    notes: list[str] = []
    calls: list[LLMResult] = []

    # -- STEP 1 : PDF -> source text -------------------------------------
    ing = ingest.ingest(pdf_path)
    notes += [f"[ingest] {n}" for n in ing.notes]
    calls += ing.vision_calls

    # -- STEP 2 : source text -> fields --------------------------------
    ext = extract.extract_fields(ing.text)
    notes += [f"[extract] {n}" for n in ext.notes]
    calls += ext.llm_calls

    # -- STEP 3 : deterministic checks --------------------------------
    problems = verify.verify(ext.fields, ing.text)
    fields = ext.fields

    # -- STEP 3b : targeted re-ask for the fields that failed --------
    rk = reask.run(fields, ing.text, problems)
    fields = rk.fields
    problems = rk.problems
    calls += rk.llm_calls
    notes += [f"[reask] {n}" for n in rk.notes]
    for r in rk.rounds:
        notes.append(
            f"[reask] round {r.round_no}: asked {r.asked} -> "
            f"accepted {r.accepted or '[]'}"
            + (f", rejected {r.rejected}" if r.rejected else "")
        )

    # -- STEP 4 : Invoice -> TEIF XML -> validate ---------------------
    invoice = assemble.to_invoice(
        fields,
        source_text=ing.text,
        model_used=ext.model_used + ("+reask" if rk.changed else ""),
        notes=[n for n in notes],
    )
    xml, build_warnings = build_teif_xml(invoice)
    xml_findings = validate_teif_xml(xml)

    # -- verdict -------------------------------------------------------
    needs_review = (
        verify.needs_human_review(problems)
        or any(getattr(f, "severity", "") == "error" for f in xml_findings)
        or ing.low_confidence_source
        or bool(rk.still_deficient)
    )
    if ing.low_confidence_source:
        notes.append("[verdict] source is a scan / OCR-layer page -> review required "
                     "regardless of checks.")
    if rk.still_deficient:
        notes.append("[verdict] re-ask loop exhausted retries on: "
                     + ", ".join(rk.still_deficient))

    return PipelineResult(
        pdf_path=pdf_path,
        source_text=ing.text,
        classification=ing.classification,
        low_confidence_source=ing.low_confidence_source,
        fields=fields,
        text_model_used=ext.model_used,
        escalated=ext.escalated,
        problems=problems,
        reask_rounds=rk.rounds,
        reask_attempts=dict(rk.attempts),
        still_deficient=rk.still_deficient,
        invoice=invoice,
        xml=xml,
        build_warnings=build_warnings,
        xml_findings=xml_findings,
        needs_review=needs_review,
        notes=notes,
        llm_calls=calls,
    )
