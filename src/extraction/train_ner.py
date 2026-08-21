# -*- coding: utf-8 -*-
"""
Train a custom NER on top of spaCy's PRETRAINED French pipeline
(fr_core_news_sm), instead of an empty spacy.blank("fr") model.

Why this matters: a blank model starts with every weight randomly
initialized -- it has zero prior knowledge of French. It can only ever
recognize patterns literally present in your training examples, which
is why more "templates" alone has diminishing returns for generalizing
to invoices with company names / phrasing never seen in training.

fr_core_news_sm ships with a tok2vec component pretrained on a large
real French corpus. Its word representations already encode general
language structure. Fine-tuning a new NER label set on TOP of that
tok2vec gives the model a real head start at generalizing.

FIXES vs the previous version:
1. Entity spans are aligned to token boundaries via doc.char_span()
   before being handed to Example.from_dict(). Previously,
   Example.from_dict() would silently misalign/drop any span that
   didn't land exactly on a token boundary (spaCy's [W030] warning),
   which is why VAT_RATE (short spans like "7" glued to "%") and
   PHONE (spans starting with "+") were scoring 0% despite having
   plenty of training support.
2. Dropped/misaligned spans are now counted and reported explicitly,
   on their own printed lines -- not overwritten by the `\r` progress
   line, which was silently hiding this exact problem before.
3. The NER component is now a FRESH pipe, not the pretrained one.
   Reusing fr_core_news_sm's existing 'ner' component (which already
   has PER/LOC/ORG/MISC) means every training example implicitly
   teaches "this span is NOT ORG/PER" for anything labeled with your
   custom tags instead -- catastrophic interference with the
   pretrained labels. We keep the pretrained tok2vec (via
   resume_training on the shared vocab/vectors) but train a NER head
   from scratch on your label set only.

Install the pretrained pipeline once, before running this script:
    python -m spacy download fr_core_news_sm
"""

import random
from collections import Counter

import spacy
from spacy.training import Example
from spacy.util import minibatch
from thinc.schedules import compounding
from training_data import TRAIN_DATA


def make_aligned_example(nlp, text, annotations, drop_counter):
    """Build an Example with entity spans snapped to token boundaries.

    Returns None if the example ends up with zero usable entities
    (rare, but possible if every span in that example was
    unrecoverable) so it can be skipped from the batch instead of
    silently training on an unlabeled doc.
    """
    doc = nlp.make_doc(text)
    aligned_ents = []

    for start, end, label in annotations["entities"]:
        # "contract" shrinks the span inward to the nearest token
        # boundaries (safer default for numeric fields glued to
        # punctuation, e.g. "7%" -> keeps "7"). If that fails, try
        # "expand" as a fallback before giving up on the span.
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            drop_counter[label] += 1
            continue
        aligned_ents.append(span)

    # spaCy requires non-overlapping entities; filter_spans keeps the
    # longest non-overlapping set, needed because "contract"/"expand"
    # can occasionally cause two originally-distinct spans to collide.
    aligned_ents = spacy.util.filter_spans(aligned_ents)

    example_dict = {"entities": [(s.start_char, s.end_char, s.label_) for s in aligned_ents]}
    return Example.from_dict(doc, example_dict)


def train(output_dir="ner_model", n_iter=30, base_model="fr_core_news_sm"):
    print(f"Loading pretrained pipeline: {base_model}")
    nlp = spacy.load(base_model)

    # Remove the pretrained NER component (with its PER/LOC/ORG/MISC
    # labels) and add a fresh one, so our custom labels don't fight
    # the old ones. The pretrained tok2vec stays in the pipeline and
    # keeps its weights -- only the NER head is new.
    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")
    ner = nlp.add_pipe("ner", last=True)

    labels = sorted({ent[2] for _, ann in TRAIN_DATA for ent in ann["entities"]})
    for label in labels:
        ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        # resume_training() only re-uses weights for components that
        # ALREADY have an initialized shape (like the pretrained
        # tok2vec). It does NOT build one for a brand-new pipe -- our
        # fresh 'ner' component has no transition system (no moves,
        # not even the basic "O"/outside transition) until we
        # explicitly initialize it. That's what caused [E022]
        # "Could not find a transition with the name 'O'": nlp.update()
        # tried to use moves that were never built.
        #
        # ner.initialize(...) builds the component's model/moves from
        # a sample of real examples, WITHOUT touching other pipes --
        # unlike nlp.initialize(), which would also reset the
        # pretrained tok2vec we're trying to keep.
        init_drop_counter = Counter()
        init_examples = []
        for text, annotations in TRAIN_DATA:
            example = make_aligned_example(nlp, text, annotations, init_drop_counter)
            if example is not None and len(example.reference.ents) > 0:
                init_examples.append(example)

        ner.initialize(lambda: init_examples, nlp=nlp)

        # resume_training() now just gives us an optimizer, and keeps
        # the pretrained tok2vec weights intact while fine-tuning
        # everything (tok2vec + the newly-initialized ner) together.
        optimizer = nlp.resume_training()

        for i in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            drop_counter = Counter()
            n_examples = 0

            batches = list(minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001)))
            for b_idx, batch in enumerate(batches):
                examples = []
                for text, annotations in batch:
                    example = make_aligned_example(nlp, text, annotations, drop_counter)
                    if example is not None and len(example.reference.ents) > 0:
                        examples.append(example)
                        n_examples += 1

                if examples:
                    nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
                print(f"  epoch {i + 1}/{n_iter} batch {b_idx + 1}/{len(batches)}", end="\r")

            print(f"Iteration {i + 1}/{n_iter} - Loss: {losses}" + " " * 20)
            if drop_counter:
                print(
                    f"  [alignment] dropped {sum(drop_counter.values())} spans "
                    f"across {n_examples} examples this epoch: {dict(drop_counter)}"
                )

    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train()
