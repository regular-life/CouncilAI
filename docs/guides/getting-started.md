# Getting Started with CouncilAI

This guide will walk you through setting up CouncilAI locally.

## Prerequisites

* **Docker and Docker Compose** (Required)
* **LLM Access Option A: Online APIs** (Optional): An [OpenRouter](https://openrouter.ai/) and/or [Gemini](https://aistudio.google.com/apikey) API key. (Allows native web-search grounding).
* **LLM Access Option B: Local Models** (Optional): A GPU-enabled system to run local vLLM models (making the stack **100% offline and free of API fees**, with local scraping fallback for web-search).

## 1. Clone and Configure

First, clone the repository and run the setup script:

```bash
git clone https://github.com/regular-life/CouncilAI
cd CouncilAI
./setup.sh
```

The interactive `./setup.sh` script automatically:
* Copies `.env.example` to `.env` (preserving any existing keys).
* Generates a high-entropy cryptographically secure random `JWT_SECRET`.
* Prompts you to optionally input your Gemini, OpenRouter, and NVIDIA NIM keys. (These are only required if using Option A; for fully local setups, you can leave these blank).

*(Alternatively, you can manually copy `.env.example` to `.env` and fill in the values.)*

## 2. Run the Stack

To start the standard stack using cloud APIs:

```bash
docker compose up --build
```

### Running Local Models (vLLM)
If you configure any agent in `config.yaml` to use provider `local` (e.g. `provider: local`), local vLLM model execution is **automatically enabled**. Simply start the services using the `local-models` docker profile:
```bash
docker compose --profile local-models up --build
```

This will spin up the following containerized services:
- **Go backend** at `http://localhost:8080`
- **Python RAG** at `http://localhost:8000` (internal)
- **Redis** at port `6379`
- **Prometheus** at `http://localhost:9091`
- **Grafana** at `http://localhost:3000` (login: `admin` / `admin`)

## 3. Interact via the Streamlit UI

CouncilAI ships with a demo Streamlit frontend. Run it in a separate terminal (requires Python installed on your host):

```bash
pip install streamlit requests
streamlit run streamlit/app.py
```

It opens at `http://localhost:8501`. Log in with the default test credentials `demo` / `demo123`, upload a PDF, and start asking questions.

## 4. Interact via the API (cURL)

You can bypass the UI and interact directly with the high-performance Go control plane.

```bash
# Get an auth token
TOKEN=$(curl -s http://localhost:8080/api/v1/login \
  -d '{"username":"demo","password":"demo123"}' | jq -r .token)

# Upload a document
curl http://localhost:8080/api/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@your_document.pdf"

# Ask a question (Grounded on document)
curl http://localhost:8080/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key concepts?","doc_id":"your_doc_id"}'
```

See [docs/api/rest-api.md](../api/rest-api.md) for the full API reference.

## 5. Teardown

To shut down the system and preserve data:
```bash
docker compose down
```

To shut down and wipe all cached data, ChromaDB vectors, and monitoring data:
```bash
docker compose down -v
```
