import random

import spacy
from spacy.training import Example
from spacy.util import compounding, minibatch
from training_data import TRAIN_DATA


def train(output_dir="ner_model", n_iter=30):
    nlp = spacy.blank("fr")
    ner = nlp.add_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for i in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    examples.append(Example.from_dict(doc, annotations))
                nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
            print(f"Iteration {i + 1}/{n_iter} - Loss: {losses}")

    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train()
