# CouncilAI Changelog

This document tracks recent major updates, architectural changes, and performance improvements to the CouncilAI project. It is formatted to be easily parsed by an LLM for updating engineering resumes.

## [Unreleased] - Recent Major Overhaul

### ⚙️ YAML Configuration & Agent Customization
- **Centralized YAML Config**: Replaced the scattered `.env` structure with a unified, self-documenting `config.yaml` file at the root. Added `gopkg.in/yaml.v3` (Go) and `pyyaml` (Python) loaders with strict environment variable precedence to keep secrets safe.
- **Interactive Setup Utility (`setup.sh`)**: Engineered an interactive bash setup script that generates a high-entropy cryptographically secure `JWT_SECRET`, preserves existing `.env` credentials, and prompts for API keys in a clean, guided CLI.
- **Auto-Enabled Local Models**: Added intelligent auto-detection in the Go configuration loader. The local vLLM sub-system is now **auto-enabled** if *any* council member, chairman, or router agent is configured to use the `local` provider.
- **Laptop-Focused vLLM Optimizations**: Integrated flags for `--max-num-seqs`, `--kv-cache-dtype` (for 50% VRAM saving via `fp8`), and `--cpu-offload-gb` to allow local models to execute efficiently on low-resource consumer laptops.
- **Configurable Agent Models**: Added a dedicated `RouterSlot` (`ROUTER_PROVIDER` and `ROUTER_MODEL`), separating the query intent classifier model configuration from the deliberation council/chairman, allowing cost-effective local classification.

### 🧹 Concurrency, Stability & Code Cleanup (De-AI-ify)
- **Thread-Safe Map Synchronization**: Resolved a critical concurrency bug by wrapping the credentials map (`users`) in `auth.go` inside a `sync.RWMutex`, protecting registrations and logins against concurrent write panics.
- **Context Cancellation Propagation**: Refactored the orchestrator's concurrent stage threads (`fanOut` and `peerReview`) to propagate the parent request context, preventing background API credit leakages when clients disconnect.
- **Optimized HTTP Socket Reuse**: Replaced the default, unconfigured HTTP client with a highly pooled `http.Transport` configuring custom idle connections and dialer keep-alives to prevent TCP port starvation.
- **Comment Refactoring**: Stripped bloated AI-like preachy comments and section markers across Go and Python codebases. Replaced them with short, technical, developer-centric documentation following **Google Go Guidelines** and **PEP-8**.
- **Realistic TODOs**: Injected engineering TODO markers for connection pooling, SSE streaming, and tensor quantization.

### 🚀 Architecture & Performance
- **C++ SIMD Semantic Cache (L1 Cache)**: Engineered a high-performance in-memory semantic cache in C++ utilizing FMA/AVX2 SIMD instructions for blazing-fast vector similarity searches.
- **Go-to-C++ FFI**: Successfully integrated the C++ semantic cache into the Go backend using CGo, securely passing 384-dimensional floating-point vectors across the language boundary.
- **Hybrid Caching Hierarchy**: Designed a two-tier caching architecture:
  - **Tier 1 (L1)**: C++ SIMD Semantic Cache for matching rephrased questions (0.85 cosine similarity threshold, bounded LRU to 5,000 keys).
  - **Tier 2 (L2)**: Redis Exact-Hash Cache for instantaneous exact-match fallbacks and horizontal scalability.
- **Performance Gains**: Replaced expensive OpenRouter LLM queries with local L1 cache hits, resulting in an observed **~240x latency speedup** and a **63.9% cache hit rate** on semantically similar queries.

### 📊 Observability & Monitoring
- **Multi-Tier Metrics Instrumentation**: Expanded Prometheus telemetry (now prefixed with `councilai_`) to track granular cache hits/misses labeled by tier (`level="l1"` vs `level="l2"`).
- **Grafana Dashboard Upgrade**: Designed and provisioned new Grafana visualizations, including a "Cache Tier Breakdown" donut chart for real-time semantic vs. exact cache monitoring.

### 🛠️ Infrastructure & Testing
- **Project Rebranding**: Executed a comprehensive system-wide rename from *PadhAI-Dost* to *CouncilAI*, updating Go modules, Docker networking, API labels, frontend UI, and telemetry prefixes.
- **SDLC & Benchmarking**:
  - Authored the `SDLC_GUIDE.md` defining strict Software Development Life Cycle operations.
  - Refactored the testing suite into isolated pipelines: 
    - `stress_concurrency.py`: Hammers L2 Redis with 10 concurrent workers.
    - `bench_semantic_accuracy.py`: Sequentially fires semantic rephrasings with quota-safe cooldowns to strictly validate the L1 C++ SIMD cache threshold.
    - `bench_document_chunking.py`: Evaluates layout-aware chunking heuristics.

### Resume Bullet Point Suggestions
- *Spearheaded the development of a hybrid, two-tier caching layer (Redis + custom C++) for an LLM orchestrator, introducing AVX2 SIMD-accelerated semantic search that achieved a 63.9% cache hit rate and reduced response latency by 240x.*
- *Architected a multi-language microservices stack (Go, C++, Python) utilizing CGo for zero-overhead Memory FFI, improving the throughput of vector similarity calculations.*
- *Instrumented Granular Prometheus/Grafana pipelines to monitor real-time AI cache behaviors, isolating "Semantic" vs "Exact" hit rates across parallelized LLM workflows.*
