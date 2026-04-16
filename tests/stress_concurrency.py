#!/usr/bin/env python3
import requests
import time
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8080"
PDF_PATH = Path.home() / "Downloads" / "sem7 slides" / "ADL" / "11 - WGAN.pdf"

USERNAME = "loadtest"
PASSWORD = "loadtest123"

SEED_QUERIES = [
    {
        "endpoint": "/api/v1/query",
        "payload": {
            "question": "Please explain IS, FID, MMD, and how they are used for evaluation of GANs",
        },
    },
    {
        "endpoint": "/api/v1/query",
        "payload": {
            "question": "What is the Wasserstein distance and why is it better than JS divergence for training GANs?",
        },
    },
    {
        "endpoint": "/api/v1/query",
        "payload": {
            "question": "Describe the WGAN training algorithm and the role of the critic network",
        },
    },
]

CONCURRENT_WORKERS = 10
REQUESTS_PER_QUERY = 50  # per seed query, on cached path

# ── Helpers ───────────────────────────────────────────────────────────


def timed_request(method, url, **kwargs):
    """Make a request and return (response, elapsed_seconds)."""
    start = time.perf_counter()
    resp = method(url, **kwargs)
    elapsed = time.perf_counter() - start
    return resp, elapsed


def print_header(title):
    print(f"\n[{title}]")


def print_stats(label, latencies, errors):
    if not latencies:
        print(f"  {label}: no successful requests")
        return
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"  {label}")
    print(f"    Requests:  {len(latencies)} ok, {errors} errors")
    print(f"    Latency:   avg={statistics.mean(latencies):.3f}s  "
          f"p50={p50:.3f}s  p95={p95:.3f}s  p99={p99:.3f}s")
    print(f"    Min/Max:   {min(latencies):.3f}s / {max(latencies):.3f}s")
    print(f"    Throughput: {len(latencies) / sum(latencies):.1f} req/s (serial-equivalent)")


# ── Steps ─────────────────────────────────────────────────────────────


def step_login():
    """Login or register, return auth headers."""
    print_header("Step 1: Authentication")

    # Try login first
    resp, elapsed = timed_request(
        requests.post, f"{API_BASE}/api/v1/login",
        json={"username": USERNAME, "password": PASSWORD},
    )
    if resp.status_code == 200:
        token = resp.json()["token"]
        print(f"  Logged in as '{USERNAME}' ({elapsed:.3f}s)")
        return {"Authorization": f"Bearer {token}"}

    # Register
    resp, elapsed = timed_request(
        requests.post, f"{API_BASE}/api/v1/register",
        json={"username": USERNAME, "password": PASSWORD},
    )
    if resp.status_code == 201:
        token = resp.json()["token"]
        print(f"  Registered and logged in as '{USERNAME}' ({elapsed:.3f}s)")
        return {"Authorization": f"Bearer {token}"}

    print(f"  ✗ Auth failed: {resp.status_code} {resp.text}")
    sys.exit(1)


def step_ingest(headers):
    """Ingest the PDF document, return doc_id."""
    print_header("Step 2: Document Ingestion")

    if not PDF_PATH.exists():
        print(f"  ✗ PDF not found: {PDF_PATH}")
        sys.exit(1)

    with open(PDF_PATH, "rb") as f:
        resp, elapsed = timed_request(
            requests.post, f"{API_BASE}/api/v1/ingest",
            headers=headers,
            files={"file": (PDF_PATH.name, f)},
        )

    if resp.status_code == 200:
        data = resp.json()
        doc_id = data["doc_id"]
        print(f"  Ingested '{PDF_PATH.name}' → {data['chunk_count']} chunks ({elapsed:.3f}s)")
        print(f"  doc_id: {doc_id}")
        return doc_id

    print(f"  ✗ Ingestion failed: {resp.status_code} {resp.text}")
    sys.exit(1)


def step_seed(headers, doc_id):
    """Run each seed query once (hits LLMs, populates cache). Returns payloads."""
    print_header("Step 3: Seeding Cache (LLM calls — will be slow)")

    payloads = []
    for i, seed in enumerate(SEED_QUERIES, 1):
        payload = {**seed["payload"], "doc_id": doc_id}
        payloads.append((seed["endpoint"], payload))

        print(f"  [{i}/{len(SEED_QUERIES)}] {payload['question'][:70]}...")
        resp, elapsed = timed_request(
            requests.post, f"{API_BASE}{seed['endpoint']}",
            headers=headers, json=payload, timeout=300,
        )
        status = "✓ cached" if resp.status_code == 200 else f"✗ {resp.status_code}"
        cache_hit = resp.json().get("cache_hit", False) if resp.status_code == 200 else False
        if cache_hit:
            status = "✓ already cached"
        print(f"         {status} ({elapsed:.1f}s)")

    return payloads


def step_load_test(headers, payloads):
    """Hammer cached queries concurrently and collect stats."""
    print_header(f"Step 4: Load Test ({REQUESTS_PER_QUERY} × {len(payloads)} queries, {CONCURRENT_WORKERS} workers)")

    all_latencies = []
    all_errors = 0
    per_query_stats = []

    for endpoint, payload in payloads:
        query_label = payload["question"][:60]
        latencies = []
        errors = 0

        def fire(_):
            resp, elapsed = timed_request(
                requests.post, f"{API_BASE}{endpoint}",
                headers=headers, json=payload, timeout=30,
            )
            return resp.status_code, elapsed

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as pool:
            futures = [pool.submit(fire, i) for i in range(REQUESTS_PER_QUERY)]
            for f in as_completed(futures):
                status, elapsed = f.result()
                if status == 200:
                    latencies.append(elapsed)
                else:
                    errors += 1
        wall_time = time.perf_counter() - start

        all_latencies.extend(latencies)
        all_errors += errors
        per_query_stats.append((query_label, latencies, errors, wall_time))

    # Print per-query stats
    print()
    for label, lats, errs, wall in per_query_stats:
        print(f"  Query: \"{label}...\"")
        print(f"    {len(lats)} ok, {errs} errors in {wall:.2f}s wall time")
        if lats:
            print(f"    avg={statistics.mean(lats):.3f}s  "
                  f"p50={statistics.median(lats):.3f}s  "
                  f"p95={sorted(lats)[int(len(lats) * 0.95)]:.3f}s")
            print(f"    Effective throughput: {len(lats) / wall:.1f} req/s")
        print()

    # Aggregate stats
    print_header("Aggregate Results")
    total = len(all_latencies) + all_errors
    print(f"  Total requests: {total}")
    print(f"  Successful:     {len(all_latencies)}")
    print(f"  Errors:         {all_errors}")
    if all_latencies:
        s = sorted(all_latencies)
        print(f"  Latency avg:    {statistics.mean(s):.3f}s")
        print(f"  Latency p50:    {statistics.median(s):.3f}s")
        print(f"  Latency p95:    {s[int(len(s) * 0.95)]:.3f}s")
        print(f"  Latency p99:    {s[int(len(s) * 0.99)]:.3f}s")
        print(f"  Min / Max:      {s[0]:.3f}s / {s[-1]:.3f}s")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("--- CouncilAI Concurrency Stress Test ---")

    headers = step_login()
    doc_id = step_ingest(headers)
    payloads = step_seed(headers, doc_id)
    step_load_test(headers, payloads)

    print(f"\n{'─' * 60}")
    print("  Done.")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
