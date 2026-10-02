"""Benchmark the naive vs optimized serving paths against a running server.

Usage:
    uvicorn app.main:app --port 8000          # in another terminal
    python benchmarks/benchmark.py            # prints real before/after numbers
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

SUBJECTS = ["the service", "this product", "the app", "customer support", "the interface"]
ADJECTIVES = ["great", "terrible", "fast", "slow", "amazing", "buggy", "reliable", "poor"]


def make_workload(n: int, unique: int, seed: int) -> list[str]:
    """n requests drawn from `unique` distinct texts with a skewed (realistic) repeat pattern."""
    rng = random.Random(seed)
    pool = [f"{rng.choice(SUBJECTS)} was {rng.choice(ADJECTIVES)} on order {i}"
            for i in range(unique)]
    weights = [1 / (i + 1) for i in range(unique)]
    return rng.choices(pool, weights=weights, k=n)


def pct(sorted_vals: list[float], p: float) -> float:
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def timed_post(client: httpx.Client, url: str, payload: dict) -> float:
    t = time.perf_counter()
    r = client.post(url, json=payload)
    r.raise_for_status()
    return (time.perf_counter() - t) * 1000


def summarize(latencies: list[float], wall_s: float, items: int) -> dict:
    s = sorted(latencies)
    return {
        "requests": len(s),
        "items": items,
        "wall_seconds": round(wall_s, 3),
        "items_per_second": round(items / wall_s, 1),
        "mean_ms": round(sum(s) / len(s), 2),
        "p50_ms": round(pct(s, 50), 2),
        "p95_ms": round(pct(s, 95), 2),
    }


def run_single(client, url, texts, workers):
    for t in texts[:20]:  # warm-up
        timed_post(client, url, {"text": t})
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lats = list(ex.map(lambda t: timed_post(client, url, {"text": t}), texts))
    return summarize(lats, time.perf_counter() - start, len(texts))


def run_batch(client, url, texts, batch_size, workers):
    chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    timed_post(client, url, {"texts": chunks[0]})  # warm-up
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lats = list(ex.map(lambda c: timed_post(client, url, {"texts": c}), chunks))
    return summarize(lats, time.perf_counter() - start, len(texts))


def improvement(old: float, new: float, lower_is_better: bool) -> float:
    change = (old - new) / old if lower_is_better else (new - old) / old
    return round(change * 100, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--unique", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    texts = make_workload(a.requests, a.unique, a.seed)
    limits = httpx.Limits(max_connections=a.workers * 2)
    with httpx.Client(base_url=a.base_url, timeout=60, limits=limits) as c:
        c.get("/health").raise_for_status()
        naive = run_single(c, "/predict/naive", texts, a.workers)
        opt = run_single(c, "/predict", texts, a.workers)
        batch = run_batch(c, "/predict/batch", texts, a.batch_size, a.workers)

    results = {
        "config": vars(a),
        "naive": naive,
        "optimized": opt,
        "batch": batch,
        "improvement_vs_naive": {
            "optimized_p95_latency_reduction_pct":
                improvement(naive["p95_ms"], opt["p95_ms"], True),
            "optimized_throughput_increase_pct":
                improvement(naive["items_per_second"], opt["items_per_second"], False),
            "batch_throughput_increase_pct":
                improvement(naive["items_per_second"], batch["items_per_second"], False),
        },
    }

    print(f"\n{'path':<12}{'req':>6}{'items/s':>11}{'mean ms':>10}{'p50 ms':>9}{'p95 ms':>9}")
    for name, r in (("naive", naive), ("optimized", opt), ("batch", batch)):
        print(f"{name:<12}{r['requests']:>6}{r['items_per_second']:>11}"
              f"{r['mean_ms']:>10}{r['p50_ms']:>9}{r['p95_ms']:>9}")
    print("\nImprovement vs naive:")
    for k, v in results["improvement_vs_naive"].items():
        print(f"  {k}: {v}%")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
