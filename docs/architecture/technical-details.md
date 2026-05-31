# CouncilAI: Multi-Agent Deliberation & Self-Reflective Document Engine

This document provides a deep dive into the technical details of the **CouncilAI** architecture, focusing on the Go control plane, the multi-agent coordination loops, the C++ SIMD-accelerated cache, and the Python RAG pipeline.

---

## 1. Go Control Plane (Backend)

The backend is built in Go 1.22, chosen for its superior concurrency model and performance in I/O-bound tasks like orchestrating multiple parallel LLM calls.

### Concurrency Model: The Council Pattern
The core `council` package uses goroutines and `sync.WaitGroup` to execute the multi-stage orchestration:
*   **Stage 1 (Fan-out)**: N goroutines are spawned in parallel to call N different models.
*   **Stage 2 (Peer Review)**: Once Stage 1 finishes, another N goroutines are spawned to let models rank the collected candidate answers.
*   **Stage 3 (Synthesis)**: A "Chairman" model receives all answers and reviews to produce the final synthesis.
*   **Cancellation Propagation**: Active goroutine context handles are bound to the parent request context via `context.WithTimeout(ctx, o.stageTimeout)`. If a client disconnects or times out, the cancellation signal propagates instantly, tearing down background LLM sockets and preserving API key quotas.

### Memory Map Thread-Safety & Authentication
*   **Credentials Locking**: User registrations and logins are handled in-memory for demo purposes. To prevent concurrent map write runtime crashes, a `sync.RWMutex` serializes all dictionary writes and protects read access.
*   **Token Generation**: Issues standard HS256-signed JWT credentials with configurable lifespan.

### HTTP Connection Pooling
*   **Optimized Transport**: Handler communications with the Python RAG service utilize a custom `http.Transport` configuring `MaxIdleConns: 100`, `MaxIdleConnsPerHost: 100`, and `IdleConnTimeout: 90s` to pool active sockets and eliminate TCP port starvation under high load.

### CGo & Memory Management (L1 Cache)
The **Semantic Cache (L1)** is implemented in C++ and bridged to Go via CGo.
*   **FFI Boundary**: Vectors (float32 arrays) and strings (answers/docIDs) are passed across the boundary.
*   **Memory Safety**: The C++ layer implements a strict LRU (Least Recently Used) cache bounded to 5,000 entries. This prevents the Go process from experiencing memory bloat while still benefiting from C++'s predictable memory management.

### Unified Configuration Management
The configuration system is centralized via `config.yaml` located in the root of the project.
*   **Docker Volumes**: Mounted read-only into `/app/config.yaml` inside the containers.
*   **Precedence**: Environment variables automatically overlay the YAML settings. This allows general defaults (timeouts, model counts) to stay in the YAML configuration while keeping API keys and environment overrides dynamic and secure.
*   **Interactive Setup (`setup.sh`)**: A root-level bash script enables easy environment initialization, generating high-entropy secure `JWT_SECRET` strings and prompting for provider API keys in a clean CLI.

---

## 2. C++ SIMD Semantic Cache

The semantic cache is the high-performance heart of CouncilAI, localized in `services/go-backend/internal/cache/fastcache/`.

### AVX2 / FMA Optimizations
To achieve sub-millisecond similarity matching across thousands of vectors, the cache uses Intel Intrinsics:
*   **`_mm256_fmadd_ps`**: Used for the dot product and norm calculations. It performs fused multiply-add operations on 8 floats simultaneously.
*   **Performance**: Vector similarity matching for 384-dimensional vectors is roughly **240x faster** than re-invoking the RAG/LLM pipeline.

### Similarity Logic
*   **Metric**: Cosine Similarity.
*   **Threshold**: Configurable, currently tuned to **0.85**. 
*   **Process**:
    1.  Incoming text is embedded via the Python `/embed` service.
    2.  The resulting 384-dim vector is passed to the C++ cache.
    3.  The cache iterates over the `doc_id` scope and returns the best match if it exceeds the threshold.

---

## 3. Python RAG Service

The RAG service handles the "Heavy ML" and document processing tasks.

### Adaptive OCR Strategy
The system uses an **Inspection-first** strategy for PDF extraction:
1.  **Direct Extraction**: If a high-quality text layer is detected, it uses `PyPDF2`.
2.  **Layout-Aware (pdfplumber)**: If tables or complex structures are detected.
3.  **Tesseract OCR**: If the document is scanned or image-based.

### Layout-Aware Chunking
Unlike standard character-count splitters, the `chunking` module follows document semantics:
*   **Table Preservation**: Tables are kept as single units to preserve row/column context.
*   **Heading Attachment**: Headings are prepended to the subsequent paragraph to ensure the embedding captures the context.
*   **Sentence Boundaries**: Chunks are split on natural sentence boundaries using regex-based detection.

---

## 4. Web Search Grounding

If a query is made without a `doc_id`, CouncilAI provides real-time context via Web Search Grounding:
*   **API Models (Gemini)**: The Go Backend natively invokes Google Search tools using Gemini's built-in grounding features.
*   **Local Models / OpenRouter**: The system delegates to the Python RAG's `/search` endpoint, which utilizes `duckduckgo-search` and `BeautifulSoup` to scrape top web results and provide text context for the LLM.

---

## 5. Observability & Monitoring

The system is fully instrumented for production-grade monitoring.

### Prometheus Metrics
Metrics are exposed at `:8080/metrics` with the `councilai_` prefix:
*   `councilai_cache_operations_total{result="hit|miss", level="l1|l2"}`: Tracks hit rates for the semantic vs. exact caches.
*   `councilai_request_latency_seconds`: End-to-end request timing.
*   `councilai_council_response_seconds`: Granular timing for the 3-stage LLM orchestration.

### Grafana Dashboard
A pre-configured dashboard (`monitoring/grafana/dashboards/councilai.json`) is auto-provisioned to visualize:
*   Real-time Cache Hit Rate (donut chart).
*   LLM failure rates per provider.
*   p95/p99 Latency of the cached vs. non-cached paths.

---

## 6. Security Architecture

*   **Authentication**: JWT-based (HS256) with 24-hour expiry.
*   **Rate Limiting**: Sliding-window rate limiter implemented in Redis (`services/go-backend/internal/api/middleware/ratelimit.go`).
*   **Audit Logging**: Every query/ingest is logged as a structured JSON object for compliance and usage auditing.
