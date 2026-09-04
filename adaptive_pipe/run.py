# -*- coding: utf-8 -*-
"""
Command-line entry point.

    python -m adaptive_pipe.run data/docs/facture.pdf

Writes into --out (default config.OUTPUT-adjacent "data/output/adaptive_pipe/"):
    <name>.fields.json   the final fields dict (whichever tier won)
    <name>.report.txt    full report: tiers tried, memory notes, checks, verdict
    <name>.xml            ONLY written if needs_review is False -- see pipeline.py's
                          module docstring for why a needs-review document does not
                          get a filing-ready XML.

The review queue itself is a real SQLite file at config.REVIEW_DB_PATH
(default data/output/adaptive_pipe/review_queue.db) -- open it with
teif_pipeline.review.store.ReviewStore, or run
`uvicorn teif_pipeline.review.api:app` pointed at the same DB path for the
HTTP correction workflow.

Exit codes: 0 = clean XML produced · 2 = submitted for human review, no
XML yet · 1 = the run itself failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config
from .pipeline import AdaptiveResult, run as run_pipeline


def _format_report(r: AdaptiveResult) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 70)
    add(f"adaptive_pipe report  -  {os.path.basename(r.pdf_path)}")
    add("=" * 70)
    add("")
    add(f"classification    : {r.classification}"
        + ("  (LOW-CONFIDENCE SOURCE)" if r.low_confidence_source else ""))
    add(f"tier used         : {r.tier_used}  (tiers tried: {[t.tier for t in r.tiers_tried]})")
    add(f"model calls       : {len(r.llm_calls)}")
    for c in r.llm_calls:
        add(f"    {c.model:<16} {c.duration_s:6.0f}s  in={c.prompt_tokens} out={c.output_tokens}")
    add(f"total model time  : {sum(c.duration_s for c in r.llm_calls):.0f}s")
    add("")

    for t in r.tiers_tried:
        add(f"--- {t.tier} tier ---")
        if t.memory_notes:
            add("  memory (per-seller history):")
            for n in t.memory_notes:
                add(f"    - {n}")
        else:
            add("  memory: no prior history for this seller (or none applicable)")
        add(f"  errors after this tier: {t.error_count}, "
            f"still_deficient: {t.still_deficient}")
        add("")

    add("-" * 70)
    add("VERDICT: " + ("NEEDS HUMAN REVIEW" if r.needs_review else "OK - XML produced, no review needed"))
    if r.review_item_id is not None:
        add(f"  -> review_items.id = {r.review_item_id} in {config.REVIEW_DB_PATH}")
    add("-" * 70)
    add("")

    add("final deterministic checks (accurate_pipe.verify):")
    if not r.problems:
        add("    (all passed)")
    for p in r.problems:
        add(f"    {p}")
    add("")

    if r.xml_findings:
        add("TEIF XML validation:")
        for f in r.xml_findings:
            add(f"    {f!r}")
        add("")

    add("processing notes:")
    for n in r.notes:
        add(f"    {n}")
    add("")

    add("final fields:")
    add(json.dumps({k: v for k, v in r.fields.items() if not k.startswith("_")},
                   indent=2, ensure_ascii=False))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="adaptive_pipe.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="path to the invoice PDF")
    ap.add_argument("--out", default="data/output/adaptive_pipe", help="output directory")
    ap.add_argument("--quiet", action="store_true", help="don't print the report to stdout")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.pdf):
        print(f"no such file: {args.pdf}", file=sys.stderr)
        return 1

    try:
        result = run_pipeline(args.pdf)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    fields_path = os.path.join(args.out, f"{stem}.fields.json")
    report_path = os.path.join(args.out, f"{stem}.report.txt")

    with open(fields_path, "w", encoding="utf-8") as fh:
        json.dump(result.fields, fh, indent=2, ensure_ascii=False)
    report = _format_report(result)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    written = [fields_path, report_path]
    if result.xml is not None:
        xml_path = os.path.join(args.out, f"{stem}.xml")
        with open(xml_path, "w", encoding="utf-8") as fh:
            fh.write(result.xml)
        written.append(xml_path)

    if not args.quiet:
        print(report)
    print("\nwrote:\n  " + "\n  ".join(written))
    if result.needs_review:
        print(f"\nNo XML written -- submitted for review (id={result.review_item_id}). "
             f"Correct it via teif_pipeline.review, then re-run this document.")

    return 2 if result.needs_review else 0


if __name__ == "__main__":
    sys.exit(main())
