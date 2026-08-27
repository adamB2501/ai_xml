# -*- coding: utf-8 -*-
"""Shared entity-level evaluation logic for the NER pipeline.

Both train_ner.py (per-epoch dev F1, used for early stopping/checkpointing)
and evaluate_ner.py (standalone per-label report) need the exact same
"is this predicted span correct" definition. Previously they didn't: each
had its own copy, and only one of them re-aligned gold offsets onto the
tokenizer before comparing, so the two could silently disagree on the same
model. This module is the single implementation both call.
"""

from collections import defaultdict


def gold_spans(nlp, text, entities):
    """Align gold (start, end, label) character offsets onto nlp's
    tokenizer -- same contract-then-expand fallback used at training time --
    so a tokenization-boundary mismatch is never scored as a model error.
    Returns a set of (start_char, end_char, label) tuples.
    """
    doc = nlp.make_doc(text)
    spans = set()
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            continue
        spans.add((span.start_char, span.end_char, span.label_))
    return spans


def predicted_spans(nlp, text):
    doc = nlp(text)
    return {(e.start_char, e.end_char, e.label_) for e in doc.ents}


def score_dataset(nlp, dataset):
    """Micro-averaged exact-match entity precision/recall/F1 over a dataset
    of (text, {"entities": [...]}) examples."""
    tp = fp = fn = 0
    for text, ann in dataset:
        gold = gold_spans(nlp, text, ann["entities"])
        pred = predicted_spans(nlp, text)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_dataset_per_label(nlp, dataset):
    """Per-label precision/recall/F1/support plus an OVERALL row, for
    reporting (evaluate_ner.py)."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for text, ann in dataset:
        gold = gold_spans(nlp, text, ann["entities"])
        pred = predicted_spans(nlp, text)
        for span in pred:
            (tp if span in gold else fp)[span[2]] += 1
        for span in gold:
            if span not in pred:
                fn[span[2]] += 1

    labels = sorted(set(tp) | set(fp) | set(fn))
    rows = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        denom_p = tp[label] + fp[label]
        denom_r = tp[label] + fn[label]
        p = tp[label] / denom_p if denom_p else 0.0
        r = tp[label] / denom_r if denom_r else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows[label] = {"precision": p, "recall": r, "f1": f1, "support": denom_r}
        total_tp += tp[label]
        total_fp += fp[label]
        total_fn += fn[label]

    denom_p = total_tp + total_fp
    denom_r = total_tp + total_fn
    overall_p = total_tp / denom_p if denom_p else 0.0
    overall_r = total_tp / denom_r if denom_r else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0
    rows["OVERALL"] = {"precision": overall_p, "recall": overall_r, "f1": overall_f1, "support": denom_r}
    return rows


def print_report(rows):
    print(f"{'LABEL':<24}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    print("-" * 64)
    for label, m in rows.items():
        if label == "OVERALL":
            continue
        print(f"{label:<24}{m['precision']:>10.2%}{m['recall']:>10.2%}{m['f1']:>10.2%}{m['support']:>10}")
    print("-" * 64)
    m = rows["OVERALL"]
    print(f"{'OVERALL':<24}{m['precision']:>10.2%}{m['recall']:>10.2%}{m['f1']:>10.2%}{m['support']:>10}")
