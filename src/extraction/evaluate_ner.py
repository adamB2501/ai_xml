# -*- coding: utf-8 -*-
"""
Evaluates a trained spaCy NER model against held-out dev data.
Reports precision / recall / F1 per label and overall.

Usage:
    python evaluate_ner.py

Adjust MODEL_DIR and DEV_DATA import to match your project layout.
"""

from collections import defaultdict

import spacy
from spacy.training import Example

# --- adjust these two lines to match your project ---
MODEL_DIR = "ner_model"
from training_data import DEV_DATA  # or: from training_data import DEV_DATA

# ------------------------------------------------------

nlp = spacy.load(MODEL_DIR)

# per-label counts
tp = defaultdict(int)  # true positives
fp = defaultdict(int)  # false positives (predicted but wrong/not in gold)
fn = defaultdict(int)  # false negatives (in gold but missed)

for text, annotations in DEV_DATA:
    doc = nlp(text)
    gold_spans = set(annotations["entities"])  # set of (start, end, label)
    pred_spans = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)

    for span in pred_spans:
        if span in gold_spans:
            tp[span[2]] += 1
        else:
            fp[span[2]] += 1
    for span in gold_spans:
        if span not in pred_spans:
            fn[span[2]] += 1

labels = sorted(set(tp) | set(fp) | set(fn))

print(f"{'LABEL':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
print("-" * 60)

total_tp = total_fp = total_fn = 0
for label in labels:
    p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
    r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    support = tp[label] + fn[label]
    print(f"{label:<20}{p:>10.2%}{r:>10.2%}{f1:>10.2%}{support:>10}")
    total_tp += tp[label]
    total_fp += fp[label]
    total_fn += fn[label]

overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

print("-" * 60)
print(f"{'OVERALL':<20}{overall_p:>10.2%}{overall_r:>10.2%}{overall_f1:>10.2%}{total_tp + total_fn:>10}")
