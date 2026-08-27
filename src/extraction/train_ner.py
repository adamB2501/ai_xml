# -*- coding: utf-8 -*-
"""Train the invoice NER model.

Trains on a blended pool of synthetic examples spanning multiple invoice
templates (--sources), not one specific layout: the previous version of
this script trained exclusively on ttn_train_data.py (200 examples
reverse-engineered from a single real template), which is why the deployed
model only ever learned 21 of the 39 labels in the schema (no SELLER_NAME,
CURRENCY, EMAIL, BANK_NAME, IBAN, PO_NUMBER, ... -- everything that template
never happens to contain). Pooling in training_data.py's 900+ generic,
template-varied examples (plus their noise-injected twins in
noisy_training_data.py, which model the same pdfplumber extraction
artifacts -- CID glyphs, label/value decluster, spaced digits -- without
being tied to one template's layout) is what makes the result generalize.

Early stopping and checkpointing are driven by DEV_DATA -- a fixed,
independently-generated held-out set from training_data.py -- instead of a
random split carved out of whatever happened to be in the training pool.
That fixed set is also what evaluate_ner.py reports against, so "the dev
F1 that picked this checkpoint" and "the dev F1 evaluate_ner.py prints
afterward" are the same number computed the same way (see ner_eval.py).

Usage:
    python train_ner.py                              # all three sources
    python train_ner.py --sources generic noisy      # drop TTN-specific data
    python train_ner.py --max-iter 50 --patience 8
"""

import argparse
import random
from collections import Counter

import spacy
from spacy.training import Example
from spacy.util import minibatch
from thinc.schedules import compounding

from ner_eval import score_dataset
from noisy_training_data import NOISY_TRAIN_DATA
from training_data import DEV_DATA, TRAIN_DATA
from ttn_train_data import TTN_TRAIN_DATA

SOURCES = {
    "generic": TRAIN_DATA,
    "noisy": NOISY_TRAIN_DATA,
    "ttn": TTN_TRAIN_DATA,
}
DEFAULT_SOURCES = ("generic", "noisy", "ttn")


def build_train_pool(source_names):
    pool = []
    for name in source_names:
        pool.extend(SOURCES[name])
    return pool


def make_aligned_example(nlp, text, annotations, drop_counter):
    doc = nlp.make_doc(text)
    aligned_ents = []

    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            drop_counter[label] += 1
            continue
        aligned_ents.append(span)

    aligned_ents = spacy.util.filter_spans(aligned_ents)

    example_dict = {"entities": [(s.start_char, s.end_char, s.label_) for s in aligned_ents]}
    return Example.from_dict(doc, example_dict)


def train(
    output_dir="ner_model",
    sources=DEFAULT_SOURCES,
    max_iter=30,
    patience=5,
    min_delta=0.0,
    seed=2026,
):
    """max_iter is a CEILING, not a fixed count -- training stops early once
    dev F1 hasn't improved for `patience` consecutive epochs, and whatever
    gets saved to output_dir is the checkpoint from the BEST epoch seen, not
    necessarily the last one run."""
    random.seed(seed)

    train_data = build_train_pool(sources)
    dev_data = list(DEV_DATA)
    print(
        f"train examples: {len(train_data)} (sources={list(sources)})  "
        f"dev examples: {len(dev_data)} (fixed DEV_DATA, held out, never trained on)"
    )

    nlp = spacy.blank("fr")
    ner = nlp.add_pipe("ner", last=True)
    labels = sorted({ent[2] for _, ann in train_data for ent in ann["entities"]})
    for label in labels:
        ner.add_label(label)
    print(f"labels ({len(labels)}): {labels}")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        init_drop_counter = Counter()
        init_examples = []
        for text, annotations in train_data:
            example = make_aligned_example(nlp, text, annotations, init_drop_counter)
            if example is not None and len(example.reference.ents) > 0:
                init_examples.append(example)

        nlp.initialize(lambda: init_examples)
        optimizer = nlp.create_optimizer()

        best_f1 = -1.0
        best_epoch = -1
        epochs_without_improvement = 0

        for i in range(max_iter):
            random.shuffle(train_data)
            losses = {}
            drop_counter = Counter()
            n_examples = 0

            batches = list(minibatch(train_data, size=compounding(4.0, 32.0, 1.001)))
            for b_idx, batch in enumerate(batches):
                examples = []
                for text, annotations in batch:
                    example = make_aligned_example(nlp, text, annotations, drop_counter)
                    if example is not None and len(example.reference.ents) > 0:
                        examples.append(example)
                        n_examples += 1

                if examples:
                    nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
                print(f"  epoch {i + 1}/{max_iter} batch {b_idx + 1}/{len(batches)}", end="\r")

            print(f"Iteration {i + 1}/{max_iter} - Loss: {losses}" + " " * 20)
            if drop_counter:
                print(
                    f"  [alignment] dropped {sum(drop_counter.values())} spans "
                    f"across {n_examples} examples this epoch: {dict(drop_counter)}"
                )

            precision, recall, f1 = score_dataset(nlp, dev_data)
            print(f"  [dev] precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

            if f1 > best_f1 + min_delta:
                best_f1 = f1
                best_epoch = i + 1
                epochs_without_improvement = 0
                nlp.to_disk(output_dir)
                print(f"  [checkpoint] new best dev f1={f1:.3f} -- saved to {output_dir}")
            else:
                epochs_without_improvement += 1
                print(f"  [checkpoint] no improvement ({epochs_without_improvement}/{patience})")
                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping: no dev F1 improvement for {patience} epochs. "
                        f"Best f1={best_f1:.3f} at epoch {best_epoch}."
                    )
                    break

    print(f"Training finished. Best model (epoch {best_epoch}, dev f1={best_f1:.3f}) saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="ner_model")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCES),
        default=list(DEFAULT_SOURCES),
        help="Synthetic datasets to pool for training (default: generic + noisy + ttn).",
    )
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        output_dir=args.output_dir,
        sources=args.sources,
        max_iter=args.max_iter,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
    )
