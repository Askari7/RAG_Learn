# P3 — HR Policy RAG Assistant

A Retrieval-Augmented Generation assistant that answers questions about a company's HR policy PDFs, with multi-turn conversation memory, hybrid (keyword + semantic) retrieval, and token-cost monitoring. Backend is FastAPI on Vercel; frontend is Streamlit; LLM is Google Gemini.

## How it works

1. **Load** — `app/loader.py` reads every PDF under `data/` (`DirectoryLoader` + `PyPDFLoader`).
2. **Split** — `app/splitter.py` chunks the loaded documents (`RecursiveCharacterTextSplitter`, `chunk_size=300`, `chunk_overlap=50`).
3. **Index** — `app/vector_store.py` embeds the chunks with `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2-preview`) into a FAISS index (cached under `vector_store/`, rebuilt only if missing), and separately builds a `BM25Retriever` (keyword search) over the same chunks.
4. **Retrieve** — a `BM25Retriever` (k=3) and the FAISS retriever (k=3) run in parallel and are combined via `EnsembleRetriever` (Reciprocal Rank Fusion) — **hybrid search**, no reranker.
5. **Generate** — `main.py` builds a LangGraph graph: each turn retrieves fresh context for the current question, prepends the conversation's prior turns, and asks `gemini-2.5-flash` (`thinking_budget=0`) to answer grounded in that context.
6. **Remember** — a LangGraph checkpointer persists conversation state per `thread_id` (Postgres/Neon in production, SQLite locally), so follow-up questions ("does that include holidays?") resolve correctly against prior turns.
7. **Monitor** — every LLM call's token usage and estimated cost is persisted to Postgres and exposed via `GET /usage`.

## Final configuration

| Setting | Value |
|---|---|
| Chunk size / overlap | 300 / 50 |
| Retrieval | Hybrid — BM25 (`k=3`) + FAISS dense (`k=3`), combined via `EnsembleRetriever` (RRF), no reranker |
| Embedding model | `gemini-embedding-2-preview` |
| LLM | `gemini-2.5-flash`, `temperature=0.2`, `thinking_budget=0` |
| Conversation memory | LangGraph checkpointer, keyed by `thread_id` (SQLite dev / Postgres prod) |
| Deployment | FastAPI on Vercel (`api/server.py`) + Streamlit frontend (`front/streamlit.py`) |

**Measured quality** (`ragas`, averaged over 2 independent runs × 40 held-out questions): **faithfulness 0.981, answer relevancy 0.981, context recall 1.000, context precision 0.703.**

## Version history & comparison

Every configuration below was measured with `ragas` (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) on the same 40-question HR-policy eval set, each score averaged across ≥1 independent run to separate real effects from run-to-run noise (~±0.03 per metric was the typical noise floor observed for a repeated run of the *same* config).

| # | Version | Faithfulness | Relevancy | Precision | Recall | Outcome |
|---|---|---|---|---|---|---|
| 1 | Baseline — plain FAISS, `k=3`, chunk 300/50 | 0.945 | 0.942 | 0.870 | 0.902 | Working baseline |
| 2 | Chunk size sweep — 200/50 and 400/50 | 0.897–0.946 | 0.905–0.962 | 0.850–0.881 | 0.819–0.869 | 300/50 stayed the best all-around; kept |
| 3 | Local cross-encoder reranker (retrieve `k=10` → rerank → top-3) | 0.958 | 0.956 | 0.869 | 0.888 | Improved quality slightly, but `sentence-transformers`/`torch` pushed the Vercel serverless function past its **500MB bundle limit**. Reverted. |
| 4 | Cohere Rerank API (HTTP-only, no local model) | — | — | — | — | Avoided the bundle-size problem, but the trial API key's **10 calls/minute** cap made it unusable under any real load (eval runs failed even with request pacing). Reverted. |
| 5 | Hybrid search — BM25 + FAISS via `EnsembleRetriever` (RRF), no reranker | **0.981** | **0.981** | 0.703 | **1.000** | Best and most consistent result across two independent runs — perfect recall, highest faithfulness/relevancy recorded. Real, repeatable trade-off: context_precision drops ~0.17 vs. plain retrieval, because the fused candidate set is larger and less filtered. **Kept — this is the final retrieval strategy.** |
| 6 | + `thinking_budget=0`, conversation memory, cost monitoring | (same as #5) | | | | Latency/UX/observability additions layered on top of #5 without changing retrieval quality — this is the shipped version. |

**Why hybrid search over a reranker, given the reranker also scored well:** both a local cross-encoder and Cohere's hosted reranker were tried first, since reranking is the more "standard" way to fix precision. Both were ruled out for infrastructure reasons specific to this project's constraints (serverless bundle size; trial-tier rate limits) rather than quality — hybrid search reaches better quality anyway without either constraint, since it needs no extra model weights and no extra API call.

**Why the precision trade-off was accepted:** for an HR-policy assistant, a confidently wrong or incomplete answer (low faithfulness/recall) is a worse failure than a slightly noisier retrieved-context set (lower precision) that the LLM still has to reason over correctly. Hybrid search's perfect recall means the correct policy fact is essentially always retrieved; its precision cost is the one metric worth revisiting if a fast, dependency-light reranker becomes available later.

## Project structure

```
P3/
├── main.py                    # LangGraph pipeline: retrieve → generate → checkpoint → track usage
├── app/
│   ├── loader.py               # Loads PDFs from data/
│   ├── splitter.py             # Chunks documents
│   ├── vector_store.py         # Builds/loads the hybrid (BM25 + FAISS) retriever
│   ├── monitoring.py           # TokenCostEstimator: per-call token/cost accounting
│   └── usage_store.py          # Persists token usage to Postgres, serves aggregate totals
├── api/
│   ├── server.py                # FastAPI app (Vercel entry point)
│   └── routes/
│       ├── chat.py               # POST /chat  {question, thread_id} -> {response}
│       ├── health.py             # GET /health
│       └── usage.py              # GET /usage  -> cumulative token usage & cost
├── front/
│   └── streamlit.py            # Chat UI, calls the deployed API
├── data/                       # Source HR policy PDFs
├── vector_store/               # Cached FAISS index (index.faiss / index.pkl)
├── vercel.json                 # Vercel build config (api/server.py via @vercel/python)
├── requirements.txt             # Deploy-only deps, exported from pyproject.toml (`uv export --no-dev --no-hashes`)
└── pyproject.toml
```

## Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Google Gemini API key
- (Production only) a Postgres database — this project uses [Neon](https://neon.tech) via its Vercel integration

## Setup

```bash
uv sync
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-gemini-api-key

# Optional — enables persistent conversation memory & usage tracking against Postgres.
# Falls back to a local checkpoints.db (SQLite) and in-memory usage tracking if unset.
DATABASE_URL=your-neon-postgres-connection-string

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=your-project-name
```

## Running locally

```bash
# API
uv run uvicorn api.server:app --reload

# Frontend (in a separate terminal)
uv run streamlit run front/streamlit.py
```

The Streamlit app generates a `thread_id` per browser session and sends it with every request, so follow-up questions in the same session share conversation context.

## API endpoints

| Endpoint | Method | Body / Notes |
|---|---|---|
| `/chat` | POST | `{"question": str, "thread_id": str}` → `{"response": str}` |
| `/health` | GET | Liveness check |
| `/usage` | GET | `{"input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"}`, cumulative across all requests via Postgres |

## Deployment (Vercel)

`vercel.json` builds `api/server.py` as a single `@vercel/python` function. Set these in the Vercel project's environment variables:

- `GEMINI_API_KEY`
- `DATABASE_URL` — auto-populated if you attach Neon as storage via Vercel's integration (uses the pooled connection string, which is what this project expects)

`requirements.txt` is a **deploy-only** dependency list, kept separate from local dev tooling (`ragas`, `streamlit`, `uvicorn`, etc. live in `pyproject.toml`'s `dev` group instead). Regenerate it after any dependency change with:

```bash
uv export --no-dev --no-hashes --format requirements-txt --no-header -o requirements.txt
```
(`--no-hashes` matters here since the lockfile is resolved for your local machine's platform — hash-pinning would break `pip install` on Vercel's build servers.)

## Evaluation

```bash
uv run python ragas_evaluate.py
```

Runs all 40 questions in `rag_eval_dataset.json` through the pipeline (each on a fresh `thread_id`, so eval questions don't share conversation memory with each other) and scores the results with `ragas`. Detailed per-run history and reasoning behind every configuration change live in `RAG_EVALUATION_LOG.md` / `RAG_EVALUATION_LOG_40Q.md` (kept locally, not committed — see the version comparison above for the summarized findings).

## Adding documents

Drop additional PDFs into `data/`, then delete the `vector_store/` directory so the index gets rebuilt on the next run.
