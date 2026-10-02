"""FastAPI model-serving service."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request

from app.config import get_db_path, get_model_path
from app.model import ModelService, load_model, predict_texts
from app.schemas import (BatchPredictRequest, BatchPredictResponse, HistoryItem,
                         PredictRequest, PredictResponse, Prediction)
from app.storage import Store, init_db, naive_log


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = get_db_path()
    init_db(db_path)
    app.state.service = ModelService(get_model_path())
    app.state.store = Store(db_path)
    yield
    app.state.store.close()


app = FastAPI(title="AI Inference Service", version="1.0.0", lifespan=lifespan)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


@app.get("/health")
def health(request: Request):
    info = request.app.state.service.cache_info()
    return {"status": "ok", "cache_hits": info.hits, "cache_misses": info.misses}


@app.post("/predict/naive", response_model=PredictResponse)
def predict_naive(body: PredictRequest):
    """Baseline: reloads the model from disk and opens a new DB connection per request."""
    start = time.perf_counter()
    model = load_model(get_model_path())
    pred = predict_texts(model, [body.text])[0]
    naive_log(get_db_path(), body.text, pred)
    return PredictResponse(**pred, latency_ms=_ms(start))


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    """Optimized: preloaded model, LRU cache, shared WAL connection."""
    start = time.perf_counter()
    pred = request.app.state.service.predict_one(body.text)
    request.app.state.store.add_many(
        [(body.text, pred["label"], pred["confidence"])], "predict")
    return PredictResponse(**pred, latency_ms=_ms(start))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest, request: Request):
    """Optimized: one vectorized model call and one DB transaction for many texts."""
    start = time.perf_counter()
    preds = request.app.state.service.predict_batch(body.texts)
    request.app.state.store.add_many(
        [(t, p["label"], p["confidence"]) for t, p in zip(body.texts, preds)],
        "predict/batch")
    return BatchPredictResponse(
        predictions=[Prediction(**p) for p in preds], latency_ms=_ms(start))


@app.get("/history", response_model=list[HistoryItem])
def history(request: Request, limit: int = Query(20, ge=1, le=200)):
    return request.app.state.store.recent(limit)
