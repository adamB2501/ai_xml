# -*- coding: utf-8 -*-
"""
Command-line entry point.

    python -m accurate_pipe.run  data/docs/facture.pdf
    python -m accurate_pipe.run  data/docs/facture.pdf --out mydir --quiet

Writes four files into the output dir (config.OUTPUT_DIR, or --out):
    <name>.source.txt    the text STEP 1 gathered (pdfplumber + vision) --
                         the exact input the text model saw
    <name>.fields.json   the raw fields dict from the text model
    <name>.xml           the TEIF XML
    <name>.report.txt    human-readable summary: classification, timings,
                         every check result, the review verdict

Exit code:
    0  produced XML, no review needed
    2  produced XML but NEEDS HUMAN REVIEW (errors / low-confidence source)
    1  the run itself failed (bad PDF, Ollama down, ...)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config
from .pipeline import PipelineResult, run as run_pipeline


def _format_report(r: PipelineResult) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 70)
    add(f"accurate_pipe report  -  {os.path.basename(r.pdf_path)}")
    add("=" * 70)
    add("")
    add(f"classification      : {r.classification}"
        + ("  (LOW-CONFIDENCE SOURCE)" if r.low_confidence_source else ""))
    add(f"text model used     : {r.text_model_used}"
        + ("  (escalated)" if r.escalated else ""))
    add(f"model calls         : {len(r.llm_calls)}")
    for c in r.llm_calls:
        add(f"    {c.model:<16} {c.duration_s:6.0f}s  "
            f"in={c.prompt_tokens} out={c.output_tokens}")
    total_s = sum(c.duration_s for c in r.llm_calls)
    add(f"total model time    : {total_s:.0f}s")
    add("")

    add("-" * 70)
    add("VERDICT: " + ("NEEDS HUMAN REVIEW" if r.needs_review else "OK - no review needed"))
    add("-" * 70)
    add("")

    add("re-ask loop (reask.py):")
    if not r.reask_rounds:
        add("    (not triggered - first extraction passed every check)")
    for rl in r.reask_rounds:
        add(f"    round {rl.round_no} [{rl.model_seconds:.0f}s]  "
            f"asked: {rl.asked}")
        add(f"        accepted: {rl.accepted or '[]'}")
        if rl.rejected:
            add(f"        rejected: {rl.rejected}")
    if r.reask_attempts:
        add(f"    attempts per field: {r.reask_attempts}")
    if r.still_deficient:
        add(f"    STILL UNRESOLVED -> review: {r.still_deficient}")
    add("")

    add("deterministic checks (verify.py, after re-ask):")
    if not r.problems:
        add("    (all passed)")
    for p in r.problems:
        add(f"    {p}")
    add("")

    add("TEIF XML validation (teif_pipeline.validate):")
    if not r.xml_findings:
        add("    (all passed)")
    for f in r.xml_findings:
        add(f"    {f!r}")
    add("")

    if r.build_warnings:
        add("XML builder warnings:")
        for w in r.build_warnings:
            add(f"    - {w}")
        add("")

    add("processing notes:")
    for n in r.notes:
        add(f"    {n}")
    add("")

    add("-" * 70)
    add("model self-verification block:")
    add(json.dumps(r.fields.get("_verification", {}), indent=2, ensure_ascii=False))
    add("")

    add("extracted fields:")
    add(json.dumps({k: v for k, v in r.fields.items() if not k.startswith("_")},
                   indent=2, ensure_ascii=False))
    add("")

    add("-" * 70)
    add("source text handed to the text model (STEP 1 output):")
    add("-" * 70)
    add(r.source_text)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="accurate_pipe.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="path to the invoice PDF")
    ap.add_argument("--out", default=config.OUTPUT_DIR, help="output directory")
    ap.add_argument("--quiet", action="store_true", help="don't print the report to stdout")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.pdf):
        print(f"no such file: {args.pdf}", file=sys.stderr)
        return 1

    try:
        result = run_pipeline(args.pdf)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't traceback-dump
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    source_path = os.path.join(args.out, f"{stem}.source.txt")
    fields_path = os.path.join(args.out, f"{stem}.fields.json")
    xml_path = os.path.join(args.out, f"{stem}.xml")
    report_path = os.path.join(args.out, f"{stem}.report.txt")

    with open(source_path, "w", encoding="utf-8") as fh:
        fh.write(result.source_text)
    with open(fields_path, "w", encoding="utf-8") as fh:
        json.dump(result.fields, fh, indent=2, ensure_ascii=False)
    with open(xml_path, "w", encoding="utf-8") as fh:
        fh.write(result.xml)
    report = _format_report(result)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    if not args.quiet:
        print(report)
    print(f"\nwrote:\n  {source_path}\n  {fields_path}\n  {xml_path}\n  {report_path}")

    return 2 if result.needs_review else 0


if __name__ == "__main__":
    sys.exit(main())
