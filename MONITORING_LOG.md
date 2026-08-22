# Monitoring Log

Tracks infrastructure/observability additions to the P3 RAG pipeline — separate from `RAG_EVALUATION_LOG.md` / `RAG_EVALUATION_LOG_40Q.md`, which track retrieval/generation quality experiments. This file is for changes to *how the system is measured or observed*, not changes to the RAG pipeline's own behavior.

## Record 1 — 2026-08-22: Token usage & cost tracking

**What was added:**

- **`app/monitoring.py`** — `TokenUsage` (dataclass: `input_tokens`, `output_tokens`, computed `total_tokens`) and `TokenCostEstimator` (accumulates usage across calls, estimates USD cost, exposes `.report()` as a dict, `.reset()` to zero it out).
- **`main.py`** — instantiated as `token_cost_estimator = TokenCostEstimator(input_cost_per_1m=0.30, output_cost_per_1m=2.50)`. Every `llm_node` invocation pulls `usage_metadata` off the Gemini response (`input_tokens`, `output_tokens`) and feeds it into the tracker, so usage accumulates across the life of the process.
- **`api/routes/usage.py`** (new route) + registered in `api/server.py` — `GET /usage` returns `token_cost_estimator.report()`: `{input_tokens, output_tokens, total_tokens, estimated_cost_usd}`.

**Pricing used:** $0.30 / 1M input tokens, $2.50 / 1M output tokens — Gemini 2.5 Flash standard (non-batch) text pricing, verified directly against Google's own pricing page (`ai.google.dev/gemini-api/docs/pricing`) at the time this was added, not guessed. Note: Google has flagged Gemini 2.5 model retirement for October 16, 2026 — these rates and the model itself will need revisiting after that date.

**Verified working:** live end-to-end test via `TestClient` — one `/chat` call registered `{input_tokens: 240, output_tokens: 18, total_tokens: 258, estimated_cost_usd: 0.000117}` on a subsequent `GET /usage` call.

**Known limitation — in-process, in-memory only.** `token_cost_estimator` is a plain Python object living in the FastAPI process's memory:

- Resets to zero on every cold start / redeploy.
- Does **not** aggregate across concurrent serverless function instances on Vercel — each instance has its own independent counter, so `/usage` only reflects the requests that happened to land on that specific warm instance, not true cumulative production cost.
- Fine for local dev, manual testing, and eval runs (`ragas_evaluate.py` and `main.py` both run in a single process where this is fully accurate). Not sufficient for real production cost monitoring — that would need the counts persisted externally (e.g. a database, a metrics/logging service, or Vercel's own usage analytics) rather than held in-process.

**Not tracked:** embedding API calls (`GoogleGenerativeAIEmbeddings`, used for both query embedding at retrieval time and index-building) don't expose the same `usage_metadata` shape through this code path, so their token cost isn't included in `/usage`'s numbers. For this project's usage pattern (small corpus, index built once and cached, only query embeddings happen per-request) the omitted cost is minor relative to the LLM generation cost, but it means `/usage` undercounts total spend, not just LLM spend.
