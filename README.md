# AI Inference Service

A small model-serving backend (FastAPI + scikit-learn + SQLite) built to practice the
skills AI-infrastructure teams care about: serving a model behind an API, validating
input, testing, profiling, and measuring performance optimizations.

It serves a sentiment classifier through two code paths, a deliberately **naive baseline**
and an **optimized** version, plus a benchmark script that measures the difference.

## Architecture

```
client -> FastAPI (app/main.py)
            |- /predict/naive : reload model from disk + new DB connection per request   (baseline)
            |- /predict       : preloaded model + LRU cache + shared SQLite (WAL)        (optimized)
            |- /predict/batch : one vectorized model call + one DB transaction           (optimized)
            |- /history       : recent predictions from SQLite
            '- /health        : status + cache hit/miss counters
```

## Project layout

```
app/            main.py (API), model.py (inference + cache), storage.py (SQLite),
                schemas.py (validation), training.py (train model), config.py
tests/          12 pytest tests (validation, errors, batch, cache, persistence)
benchmarks/     benchmark.py (naive vs optimized, prints p50/p95 + throughput)
Dockerfile, .github/workflows/ci.yml
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.training                    # trains and saves models/sentiment.joblib
pytest -q                                 # run the tests
uvicorn app.main:app --port 8000          # start the server (API docs at /docs)
```

Try it:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "the service was excellent"}'
```

## Benchmark (get your own real numbers)

With the server running, in a second terminal:

```bash
python benchmarks/benchmark.py
```

It prints requests/sec, mean, p50 and p95 latency for each path and calculates the
improvement percentages. Results are saved to `benchmarks/results.json`.
Run it 2-3 times and use a typical result. Numbers depend on your machine, so always
quote your own.

## Profile the bottleneck yourself

Before optimizing, find where time goes:

```bash
pip install py-spy
py-spy record -o profile.svg -- uvicorn app.main:app --port 8000
# run the benchmark, stop the server, open profile.svg
```

You should see model loading (`joblib.load`) and per-request SQLite commits dominate the
naive path. That is the story behind each optimization.

## Optimizations (what changed and why)

| Problem in naive path | Fix | Where |
|---|---|---|
| Model reloaded from disk on every request | Load once at startup | `app/main.py` lifespan |
| Repeated inputs recomputed | LRU cache on single predictions | `app/model.py` |
| New DB connection + fsync commit per request | Shared connection, WAL, `synchronous=NORMAL` | `app/storage.py` |
| One model call per item | Batch endpoint: one vectorized call + one transaction | `/predict/batch` |

## Design tradeoffs

- **LRU cache:** big win on repeated inputs, but uses memory and gives no benefit for unique inputs. Cache size is configurable.
- **SQLite + WAL:** simple and fast for one node, but a single writer. Postgres would be the next step for multiple instances.
- **Batching:** best throughput, but adds latency for each individual item and requires clients to group requests.
- **Synthetic dataset:** keeps the project runnable offline, so the model's 100% accuracy is not meaningful. Swap in a real dataset (SST-2, IMDB) for a real model.

## Ideas to extend (each is a good resume line)

1. Replace the model with a Hugging Face transformer (e.g. `distilbert-base-uncased-finetuned-sst-2-english`) and re-run the benchmark.
2. Add dynamic request batching (collect requests for ~10 ms, then run one model call).
3. Add Prometheus metrics and a Grafana dashboard.
4. Deploy with the Dockerfile to AWS/GCP and load-test with `locust` or `k6`.
5. Add a Redis cache and compare against the in-process LRU cache.

## Results

Machine: MacBook Air. 1,000 requests, 16 concurrent workers, Python 3.9.

| path | items/s | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|
| naive | 915.7 | 16.72 | 9.47 | 45.24 |
| optimized | 1483.3 | 10.73 | 7.94 | 20.18 |
| batch (32/request) | 34679.1 | 12.76 | 12.09 | 20.35 |

Optimized vs naive: p95 latency reduced by 55.4%, throughput increased by 62.0%.
Batching (32 items per request) reached ~38x the naive throughput.
