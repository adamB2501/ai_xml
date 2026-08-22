# -*- coding: utf-8 -*-

import random
from collections import Counter

import spacy
from spacy.training import Example
from spacy.util import minibatch
from thinc.schedules import compounding
#from noisy_training_data import NOISY_TRAIN_DATA
#from training_data import TRAIN_DATA
from ttn_train_data import TTN_TRAIN_DATA

ALL_TRAIN_DATA = TTN_TRAIN_DATA


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


def split_train_dev(data, dev_frac=0.15, seed=2026):
    """Held-out dev split, separate from training -- required for early
    stopping to mean anything. Never trained on."""
    data = list(data)
    rng = random.Random(seed)
    rng.shuffle(data)
    n_dev = max(1, int(len(data) * dev_frac))
    return data[n_dev:], data[:n_dev]


def evaluate_f1(nlp, dataset):
    """
    Micro-averaged exact-match entity F1 on a held-out set. Gold spans are
    aligned the same way as training (char_span contract/expand) before
    comparing, so a genuine model miss isn't confused with a tokenization
    alignment artifact.
    """
    tp = fp = fn = 0
    for text, ann in dataset:
        doc = nlp.make_doc(text)
        gold_set = set()
        for start, end, label in ann["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is None:
                continue
            gold_set.add((span.start_char, span.end_char, span.label_))

        pred_doc = nlp(text)
        pred_set = {(e.start_char, e.end_char, e.label_) for e in pred_doc.ents}

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def train(output_dir="ner_model", max_iter=30, patience=5, dev_frac=0.15, min_delta=0.0):
    """
    max_iter is now a CEILING, not a fixed count -- training stops early
    once dev F1 hasn't improved for `patience` consecutive epochs, and
    whatever gets saved to output_dir is the checkpoint from the BEST
    epoch seen, not necessarily the last one run.
    """
    nlp = spacy.blank("fr")
    ner = nlp.add_pipe("ner", last=True)

    labels = sorted({ent[2] for _, ann in ALL_TRAIN_DATA for ent in ann["entities"]})
    for label in labels:
        ner.add_label(label)

    train_data, dev_data = split_train_dev(ALL_TRAIN_DATA, dev_frac=dev_frac)
    print(f"train examples: {len(train_data)}  dev examples: {len(dev_data)} (held out, never trained on)")

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

            precision, recall, f1 = evaluate_f1(nlp, dev_data)
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


if __name__ == "__main__":
    train()