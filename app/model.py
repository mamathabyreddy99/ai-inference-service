"""Model loading and inference helpers."""
from functools import lru_cache
from pathlib import Path

import joblib


def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Run: python -m app.training")
    return joblib.load(path)


def predict_texts(model, texts: list[str]) -> list[dict]:
    """Vectorized prediction for one or many texts."""
    probs = model.predict_proba(texts)
    classes = model.classes_
    out = []
    for row in probs:
        i = int(row.argmax())
        out.append({"label": str(classes[i]), "confidence": round(float(row[i]), 4)})
    return out


class ModelService:
    """Optimized path: model loaded once, repeated inputs served from an LRU cache."""

    def __init__(self, path: str, cache_size: int = 4096):
        self.model = load_model(path)
        self._cached = lru_cache(maxsize=cache_size)(self._predict_one)

    def _predict_one(self, text: str) -> tuple[str, float]:
        p = predict_texts(self.model, [text])[0]
        return p["label"], p["confidence"]

    def predict_one(self, text: str) -> dict:
        label, conf = self._cached(text)
        return {"label": label, "confidence": conf}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return predict_texts(self.model, texts)

    def cache_info(self):
        return self._cached.cache_info()
