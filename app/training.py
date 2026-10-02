"""Train a small sentiment classifier and save it to disk.

Run:  python -m app.training
The dataset is synthetic (template-generated) so the project runs offline.
Swap in a real dataset (e.g. IMDB, SST-2) to make the model meaningful.
"""
import itertools
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import get_model_path

POS = ["great", "excellent", "amazing", "fantastic", "wonderful",
       "superb", "reliable", "fast", "delightful", "impressive"]
NEG = ["terrible", "awful", "horrible", "disappointing", "slow",
       "broken", "useless", "frustrating", "poor", "buggy"]
SUBJECTS = ["the service", "this product", "the app", "customer support",
            "the documentation", "the latest update", "the interface",
            "the delivery", "this tool", "the setup process"]
TEMPLATES = ["{s} was {a}", "i found {s} to be {a}", "honestly {s} is {a}",
             "{s} felt {a} overall", "i did not expect {s} to be so {a}"]


def build_dataset():
    texts, labels = [], []
    for label, adjs in (("positive", POS), ("negative", NEG)):
        for s, a, t in itertools.product(SUBJECTS, adjs, TEMPLATES):
            texts.append(t.format(s=s, a=a))
            labels.append(label)
    return texts, labels


def train_and_save(path=None, verbose: bool = True) -> float:
    path = path or get_model_path()
    texts, labels = build_dataset()
    x_tr, x_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(x_tr, y_tr)
    acc = pipe.score(x_te, y_te)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    if verbose:
        print(f"Saved model to {path} (holdout accuracy: {acc:.3f})")
    return acc


if __name__ == "__main__":
    train_and_save()
