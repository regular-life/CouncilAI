# CouncilAI: SDLC Operations & Testing Guide

This guide defines the standard procedures for testing, benchmarking, and monitoring the **CouncilAI** hybrid caching architecture.

## 🛠️ The Test Suite

We maintain three specialized test suites to validate different layers of the SDLC (Software Development Life Cycle).

### 1. Semantic Accuracy Benchmark (`bench_semantic_accuracy.py`)
> [!IMPORTANT]
> **This is our current primary verification tool.** Use this to validate any changes to the C++ SIMD Semantic Cache or the embedding similarity margins.

*   **Goal**: Ensure that rephrased queries (e.g., "What is mode collapse?" vs. "How do modes fail?") correctly hit the L1 cache.
*   **Mode**: Sequential execution (Quota-safe).
*   **Command**:
    ```bash
    python tests/bench_semantic_accuracy.py
    ```

### 2. Concurrency Stress Test (`stress_concurrency.py`)
*   **Goal**: Validate infrastructure stability and L2 Redis throughput under high concurrent load.
*   **Mode**: Parallel execution (10 workers).
*   **Command**:
    ```bash
    python tests/stress_concurrency.py
    ```

### 3. Document Chunking Benchmark (`bench_document_chunking.py`)
*   **Goal**: Measure the performance of layout-aware chunking on large PDFs.
*   **Command**:
    ```bash
    python tests/bench_document_chunking.py
    ```

---

## 📈 Viewing Results

### 🔬 Local JSON Reports
The `bench_semantic_accuracy.py` script generates a detailed JSON report:
*   **Path**: `tests/benchmark_results.json`
*   **Contents**: Per-query latency histograms, hit-rates by topic, and end-to-end "Speedup" metrics (e.g., 240x faster than cold LLM path).

### 📊 Real-time Grafana Dashboard
Access the dashboard at **[http://localhost:3000](http://localhost:3000)** (Admin: `admin/admin`).
*   **Panel: Cache Tier Breakdown**: Shows live breakdown of L1 (SIMD) vs. L2 (Redis) hits.
*   **Panel: Request Latency**: Visualizes the micro-latency of the cache path vs. the slow LLM path.

---

## 🔍 Monitoring & Logs

### Real-time Service Logs
To debug cache misses or C++ compilation issues in the Go backend:
```bash
docker compose logs -f go-backend
```

To monitor the Python RAG /embed service:
```bash
docker compose logs -f python-rag
```

---

## 🔄 SDLC Integration
1.  **Develop**: Modify C++ CGo code or Python RAG logic.
2.  **Deploy**: `docker compose up -d --build`.
3.  **Validate**: Run `bench_semantic_accuracy.py`.
4.  **Certify**: Ensure hit-rate is > 60% for semantic variants and Speedup is > 100x.
