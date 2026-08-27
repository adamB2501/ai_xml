# -*- coding: utf-8 -*-
"""Evaluate a trained NER model two independent ways -- a good synthetic
span-F1 doesn't guarantee extract_fields() returns the right value on a
real PDF end to end (tokenization, extraction plumbing, and the grouping
logic in ner_extractor.py all sit between "model saw a good entity" and
"the right string came out"):

  --mode span   Per-label precision/recall/F1 on the fixed synthetic
                DEV_DATA set (training_data.py) -- the same set
                train_ner.py uses for early stopping/checkpointing, scored
                the same way (ner_eval.py), so this number matches what
                training printed for the checkpoint that got saved.

  --mode e2e    Runs pdf_reader.extract_text() -> ner_extractor.extract_fields()
                on the real sample PDFs in data/samples/, and compares the
                result field-by-field against data/samples/ground_truth.json.
                This is the closer proxy for "does this generalize to an
                actual invoice PDF" -- span F1 alone can't catch a bug in
                the extraction/grouping plumbing.

Usage:
    python evaluate_ner.py                  # both modes
    python evaluate_ner.py --mode span
    python evaluate_ner.py --mode e2e
    python evaluate_ner.py --model-dir other_model
"""

import argparse
import json
import os
import re

import spacy

from ner_eval import print_report, score_dataset_per_label
from training_data import DEV_DATA

# Resolved from this file's location (not CWD) so --mode e2e works
# regardless of which directory you invoke the script from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES_DIR = os.path.join(_REPO_ROOT, "data", "samples")
GROUND_TRUTH_PATH = os.path.join(SAMPLES_DIR, "ground_truth.json")

# ground_truth.json key -> extract_fields() key. A few ground-truth keys map
# to a *list*-valued extractor field (label/value fragments can legitimately
# repeat -- see ner_extractor.REPEATABLE_LABELS) rather than a singleton, so
# those are matched by "is the gold value present anywhere in that list".
SINGLETON_FIELD_MAP = {
    "invoice_number": "invoice_number",
    "issue_date": "issue_date",
    "seller_name": "seller_name",
    "buyer_name": "buyer_name",
    "total_ht": "total_ht",
    "tva_amount": "tva_amount",
    "total_ttc": "total_ttc",
    "currency": "currency",
}
LIST_FIELD_MAP = {
    "seller_tax_id": "seller_tax_ids",
    "buyer_tax_id": "buyer_tax_ids",
    "seller_address": "seller_addresses",
    "buyer_address": "buyer_addresses",
}
NUMERIC_FIELDS = {"total_ht", "tva_amount", "total_ttc"}
# NOTE: ground_truth.json's line_items[].unit (e.g. "heure", "licence") has
# no corresponding trained label -- ITEM_LABELS only covers
# description/qty/unit_price/line_total -- so it's intentionally not scored
# here rather than faked as a match.


def _norm_text(s):
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _norm_number(s):
    """Whichever of '.'/',' appears LAST in the string is the decimal
    separator; earlier ones are thousands separators to strip. Anchoring on
    "exactly 3 digits after the separator" instead breaks on TND amounts,
    which conventionally use exactly 3 decimals (millimes) -- "303.847"
    would misparse as "303847" under that rule."""
    if s is None or s == "":
        return None
    s = re.sub(r"[^\d.,\-]", "", str(s))
    if not s:
        return None
    negative = s.startswith("-")
    s = s.lstrip("-")
    sep_pos = max(s.rfind("."), s.rfind(","))
    if sep_pos == -1:
        integer_part, decimal_part = s, ""
    else:
        integer_part = re.sub(r"[.,]", "", s[:sep_pos])
        decimal_part = s[sep_pos + 1:]
    if not integer_part and not decimal_part:
        return None
    text = (integer_part or "0") + ("." + decimal_part if decimal_part else "")
    try:
        return round(float(("-" + text) if negative else text), 2)
    except ValueError:
        return None


def _field_matches(gt_key, gold_value, extracted):
    if gt_key in SINGLETON_FIELD_MAP:
        pred = extracted.get(SINGLETON_FIELD_MAP[gt_key], "")
        if gt_key in NUMERIC_FIELDS:
            pn, gn = _norm_number(pred), _norm_number(gold_value)
            return pn is not None and pn == gn
        return _norm_text(pred) == _norm_text(gold_value)
    if gt_key in LIST_FIELD_MAP:
        candidates = extracted.get(LIST_FIELD_MAP[gt_key], [])
        return any(_norm_text(c) == _norm_text(gold_value) for c in candidates)
    return False


def _line_items_match(gold_items, extracted_items):
    matched = 0
    for gold_item in gold_items:
        for ex_item in extracted_items:
            if _norm_text(ex_item.get("description") or "") == _norm_text(gold_item["description"]):
                matched += 1
                break
    return matched, len(gold_items)


def run_span_eval(model_dir):
    nlp = spacy.load(model_dir)
    print(f"=== Span-level eval on synthetic DEV_DATA ({len(DEV_DATA)} examples), model={model_dir} ===")
    rows = score_dataset_per_label(nlp, DEV_DATA)
    print_report(rows)


def run_e2e_eval(model_dir):
    # Deferred imports: both pull in ner_extractor.py, which loads a model
    # at import time via NER_MODEL_DIR -- set before importing so --model-dir
    # is actually respected here.
    os.environ["NER_MODEL_DIR"] = model_dir
    from ner_extractor import extract_fields
    from pdf_reader import extract_text

    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"[skip] no ground truth found at {GROUND_TRUTH_PATH}")
        return

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    field_hits = {k: 0 for k in list(SINGLETON_FIELD_MAP) + list(LIST_FIELD_MAP)}
    field_total = {k: 0 for k in field_hits}
    item_matched_total = 0
    item_gold_total = 0
    per_file_rows = []

    for entry in ground_truth:
        pdf_path = os.path.join(SAMPLES_DIR, entry["file"])
        if not os.path.exists(pdf_path):
            print(f"[skip] {entry['file']} not found in {SAMPLES_DIR}")
            continue

        text = extract_text(pdf_path)
        extracted = extract_fields(text)

        hits = total = 0
        for gt_key in field_hits:
            if gt_key not in entry:
                continue
            total += 1
            field_total[gt_key] += 1
            if _field_matches(gt_key, entry[gt_key], extracted):
                hits += 1
                field_hits[gt_key] += 1

        matched, gold_n = _line_items_match(entry.get("line_items", []), extracted.get("line_items", []))
        item_matched_total += matched
        item_gold_total += gold_n

        per_file_rows.append((entry["file"], hits, total, matched, gold_n))

    print(f"=== End-to-end eval on real PDFs ({SAMPLES_DIR}), model={model_dir} ===")
    print(f"{'FILE':<18}{'Fields':>12}{'Line items':>14}")
    for file, hits, total, matched, gold_n in per_file_rows:
        print(f"{file:<18}{f'{hits}/{total}':>12}{f'{matched}/{gold_n}':>14}")

    print("\nPer-field accuracy:")
    for key in field_hits:
        if field_total[key] == 0:
            continue
        acc = field_hits[key] / field_total[key]
        print(f"  {key:<16}{acc:>8.1%}  ({field_hits[key]}/{field_total[key]})")

    if item_gold_total:
        print(
            f"\nLine items matched (by description): {item_matched_total}/{item_gold_total} "
            f"({item_matched_total / item_gold_total:.1%})"
        )
    print("\n[note] 'unit' has no trained label and is not scored.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["span", "e2e", "both"], default="both")
    parser.add_argument("--model-dir", default="ner_model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode in ("span", "both"):
        run_span_eval(args.model_dir)
        if args.mode == "both":
            print()
    if args.mode in ("e2e", "both"):
        run_e2e_eval(args.model_dir)
