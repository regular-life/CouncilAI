# CouncilAI: Multi-LLM Document Engine

A document Q&A system built around a multi-LLM council. Upload a PDF, ask questions, and get answers that have been independently generated, peer-reviewed, and synthesized across multiple language models.

The council pattern is inspired by [Karpathy's LLM Council](https://github.com/karpathy/llm-council) — instead of trusting a single model, three models answer independently, review each other's work, and a chairman synthesizes the final response.

Originally built as a study tool ("PadhAI Dost" = "Study Friend" in Hindi), the architecture is general-purpose and works for any document-grounded Q&A task.

---

## Architecture

```mermaid
graph TB
    Client["Client<br/>(Streamlit / curl)"]

    subgraph Go["Go Control Plane · :8080"]
        Router["Chi Router"]
        Auth["JWT Auth"]
        RL["Rate Limiter"]
        Cache["Redis Cache"]
        Council["LLM Council<br/>(3-stage)"]
        Metrics["Prometheus<br/>/metrics"]
        Audit["Audit Logger"]
    end

    subgraph Python["Python RAG Service · :8000"]
        Inspect["Document Inspector"]
        OCR["Adaptive OCR"]
        Chunk["Layout-Aware Chunker"]
        Embed["Transformer Embeddings"]
        Chroma["ChromaDB"]
    end

    subgraph LLMs["LLM Providers"]
        M1["Council Model 1"]
        M2["Council Model 2"]
        M3["Council Model 3"]
        Chairman["Chairman<br/>(Gemini)"]
    end

    subgraph Monitoring["Monitoring"]
        Prom["Prometheus · :9091"]
        Graf["Grafana · :3000"]
    end

    Redis[("Redis · :6379")]

    Client --> Router
    Router --> Auth --> RL
    RL --> Cache
    Cache -->|miss| Council
    Council -->|retrieve chunks| Python
    Council -->|fan-out| M1 & M2 & M3
    Council -->|synthesize| Chairman
    Router --> Metrics
    Router --> Audit

    Inspect --> OCR --> Chunk --> Embed --> Chroma
    Cache --> Redis
    RL --> Redis
    Metrics -.->|scrape| Prom
    Prom -.->|visualize| Graf
```

### Request Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant G as Go Backend
    participant R as Redis
    participant P as Python RAG
    participant L as LLM Council

    C->>G: POST /api/v1/query (JWT)
    G->>G: Validate JWT + Rate Limit
    G->>R: Check cache

    alt Cache Hit
        R-->>G: Cached response
        G-->>C: 200 OK (cache_hit: true)
    else Cache Miss
        G->>P: POST /retrieve (question, doc_id)
        P->>P: Embed → ChromaDB search
        P-->>G: Top-K document chunks

        Note over G,L: Stage 1 — Individual Responses
        G->>L: Fan-out to 3 LLMs (parallel)
        L-->>G: 3 independent answers

        Note over G,L: Stage 2 — Peer Review
        G->>L: Each model ranks others' answers
        L-->>G: Rankings + reasoning

        Note over G,L: Stage 3 — Chairman Synthesis
        G->>L: Chairman sees all answers + reviews
        L-->>G: Final synthesized answer

        G->>R: Cache result
        G-->>C: 200 OK (answer, confidence, source)
    end
```

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- An [OpenRouter](https://openrouter.ai/) API key (free tier works)
- A [Gemini](https://aistudio.google.com/apikey) API key

### 1. Clone and Configure

```bash
git clone https://github.com/regular-life/PadhAI-Dost
cd PadhAI-Dost
cp .env.example .env
```

Open `.env` and add your API keys:
```bash
GEMINI_API_KEY=your-gemini-key
OPENROUTER_API_KEY=your-openrouter-key
```

### 2. Run

```bash
docker compose up --build
```

This starts five containers:
- **Go backend** at `http://localhost:8080`
- **Python RAG** at `http://localhost:8000` (internal)
- **Redis** at port `6379`
- **Prometheus** at `http://localhost:9091`
- **Grafana** at `http://localhost:3000` (login: `admin` / `admin`)

### 3. Use the Streamlit Frontend

In a separate terminal:
```bash
pip install streamlit requests
streamlit run streamlit/app.py
```

Opens at `http://localhost:8501`. Log in with `demo` / `demo123`, upload a PDF, ask questions.

### 4. Or Use the API

```bash
# Get a token
TOKEN=$(curl -s http://localhost:8080/api/v1/login \
  -d '{"username":"demo","password":"demo123"}' | jq -r .token)

# Upload a PDF
curl http://localhost:8080/api/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@your_document.pdf"

# Ask a question
curl http://localhost:8080/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key concepts?","doc_id":"your_doc_id"}'
```

---

## Project Structure

```
PadhAI-Dost/
├── docker-compose.yml
├── .env.example
│
├── services/
│   ├── go-backend/                 # Control plane (Go)
│   │   ├── cmd/server/main.go
│   │   └── internal/
│   │       ├── api/                # Router, handlers, middleware
│   │       ├── council/            # 3-stage LLM council
│   │       ├── llm/               # OpenRouter + Gemini clients
│   │       ├── auth/              # JWT authentication
│   │       ├── cache/             # Redis caching
│   │       ├── metrics/           # Prometheus counters & histograms
│   │       └── audit/             # Structured JSON audit logs
│   │
│   └── python-rag/                 # RAG service (Python/FastAPI)
│       └── app/
│           ├── main.py
│           ├── inspection/         # Document type detection
│           ├── ocr/                # Adaptive OCR routing
│           ├── chunking/           # Layout-aware chunking
│           ├── embedding/          # Transformer embeddings
│           └── retrieval/          # ChromaDB vector store
│
├── monitoring/                      # Prometheus + Grafana
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/            # Auto-provisioned datasources & dashboards
│       └── dashboards/              # Pre-built dashboard JSON
│
└── streamlit/app.py                # Demo frontend
```

---

## API Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/login` | No | Get JWT token |
| POST | `/api/v1/register` | No | Create account |
| POST | `/api/v1/query` | Yes | Ask a question |
| POST | `/api/v1/ingest` | Yes | Upload a document |
| POST | `/api/v1/explain` | Yes | Generate document explanation |
| POST | `/api/v1/generate-questions` | Yes | Generate assessment questions |
| GET | `/health` | No | Health check |
| GET | `/metrics` | No | Prometheus metrics |

---

## Logging and Monitoring

### Log Structure

The Go backend emits structured logs to stdout. Each log line is prefixed with a tag:

| Tag | What It Logs |
|-----|-------------|
| `[HTTP]` | Every request: method, path, status code, latency |
| `[Council]` | Council stage transitions and failures |
| `[Cache]` | Cache hits and misses |
| `[Audit]` | Structured JSON: user_id, doc_id, query_hash, latency, status |

The Python RAG service uses standard `uvicorn` access logs plus structured application logs via Python's `logging` module.

### Viewing Logs

```bash
# All services (follow mode)
docker compose logs -f

# Go backend only
docker compose logs -f go-backend

# Python RAG only
docker compose logs -f python-rag

# Filter for council activity
docker compose logs -f go-backend | grep "\[Council\]"

# Filter for audit entries (JSON, good for piping to jq)
docker compose logs -f go-backend | grep "\[Audit\]" | sed 's/.*\[Audit\] //' | jq .
```

### Prometheus Metrics

The Go backend exposes Prometheus metrics at `http://localhost:8080/metrics`. Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `padhai_dost_request_count_total` | Counter | Total HTTP requests (by method, path, status) |
| `padhai_dost_request_latency_seconds` | Histogram | Request latency distribution |
| `padhai_dost_council_response_seconds` | Histogram | End-to-end council latency |
| `padhai_dost_chairman_synthesis_count_total` | Counter | How often the chairman model is called |
| `padhai_dost_llm_failure_count_total` | Counter | LLM call failures |
| `padhai_dost_cache_operations_total` | Counter | Cache hits vs misses |

You can scrape this endpoint with any Prometheus-compatible tool (Grafana, Prometheus server, etc.) or just curl it:
```bash
curl -s http://localhost:8080/metrics | grep padhai_dost
```

### Grafana Dashboard

A pre-built dashboard is auto-provisioned at startup. Open `http://localhost:3000` and log in with `admin` / `admin`.

The **PadhAI-Dost** dashboard includes:
- **Request Rate** — HTTP requests per second by method/path/status
- **Request Latency** — p50, p95, p99 latency percentiles
- **Council Response Time** — end-to-end council orchestration latency
- **Cache Hit Rate** — hit vs miss donut chart
- **Counters** — total requests, chairman synthesis calls, LLM failures

---

## Configuration

All configuration is via environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required. For chairman synthesis |
| `OPENROUTER_API_KEY` | — | Required. For council models |
| `COUNCIL_MODEL_1/2/3` | See .env.example | Council LLMs via OpenRouter |
| `CHAIRMAN_MODEL` | `gemini-3-flash-preview` | Synthesis model via Gemini |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `STAGE_TIMEOUT` | `30s` | Per-stage timeout |
| `LLM_TIMEOUT` | `120s` | Overall timeout |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Control Plane | Go 1.22, Chi router, JWT, Prometheus |
| RAG Service | Python 3.11, FastAPI, LangChain, ChromaDB |
| Embeddings | BAAI/bge-small-en-v1.5 (runs locally, CPU) |
| LLM Providers | OpenRouter (free tier) + Gemini API |
| Cache | Redis 7 |
| OCR | Tesseract, pdfplumber |
| Monitoring | Prometheus, Grafana |
| Containers | Docker Compose |
| Frontend | Streamlit |

---

## Load Testing

A self-contained load test script is included:

```bash
python tests/load_test.py
```

It logs in, ingests a document, seeds 3 queries (one-time LLM calls to populate cache), then hammers the cached paths with concurrent workers. All subsequent requests are cache hits — no API rate limits burned. Prints per-query and aggregate latency stats (avg, p50, p95, p99, throughput).

---

## Stopping

```bash
docker compose down          # Stop containers
docker compose down -v       # Stop and delete all data
```
