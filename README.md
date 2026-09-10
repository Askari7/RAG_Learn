# P3 — HR Policy RAG Assistant

A Retrieval-Augmented Generation assistant that answers questions about a company's HR policy PDFs, with multi-turn conversation memory, hybrid (keyword + semantic) retrieval with reranking, corrective retrieval, and token-cost monitoring. Backend is FastAPI on Vercel; frontend is Streamlit; LLM is Google Gemini.

## How it works

The pipeline is no longer a straight line — `main.py`'s LangGraph makes a decision after every retrieval instead of always following the same path:

1. **Load** — `app/loader.py` reads every PDF under `data/` (`DirectoryLoader` + `PyPDFLoader`), merges each file's per-page text into one continuous document (so a sentence spanning a page break isn't guaranteed to be split before chunking even starts), and strips the repeated confidentiality-notice/"Issue #" boilerplate every page carries (which otherwise lands mid-sentence wherever a page break falls).
2. **Split** — `app/splitter.py` chunks the loaded documents (`RecursiveCharacterTextSplitter`, `chunk_size=350`, `chunk_overlap=100`).
3. **Index** — `app/vector_store.py` embeds the chunks with `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2-preview`) into a FAISS index (cached under `vector_store/`, rebuilt only if missing), and separately builds a `BM25Retriever` (keyword search) over the same chunks.
4. **Retrieve** — a `BM25Retriever` (k=3) and the FAISS retriever (k=3) run in parallel and are combined via `EnsembleRetriever` (Reciprocal Rank Fusion) — **hybrid search**, yielding a ~6-document candidate set.
5. **Rerank** — `app/reranker.py` scores that candidate set with `flashrank` (`ms-marco-TinyBERT-L-2-v2`, a small ONNX cross-encoder — no torch/sentence-transformers) and keeps the top 3, so the LLM sees the most relevant chunks instead of the full unfiltered union.
6. **Grade** — a cheap LLM call checks whether the reranked context actually contains enough information to answer the question. If yes, skip to Generate.
7. **Rewrite & retry** (only if the grader says no) — another LLM call rewrites the question into a better search query, and the pipeline goes back to step 4 with that query. Capped at one retry (`MAX_CORRECTIVE_RETRIES` in `main.py`), after which it proceeds to Generate regardless — this is **corrective retrieval**: the system tries to fix a bad retrieval once, then answers honestly with whatever it has rather than looping forever.
8. **Generate** — `main.py` prepends the conversation's prior turns to the (possibly corrected) context and asks `gemini-2.5-flash` (`thinking_budget=0`) to answer grounded in it.
9. **Remember** — a LangGraph checkpointer persists conversation state per `thread_id` (Postgres/Neon in production, SQLite locally), so follow-up questions ("does that include holidays?") resolve correctly against prior turns.
10. **Monitor** — every LLM call's (retrieval-side grading/rewriting included) token usage and estimated cost is persisted to Postgres and exposed via `GET /usage`.

## Final configuration

| Setting | Value |
|---|---|
| Chunk size / overlap | 350 / 100 |
| Retrieval | Hybrid — BM25 (`k=3`) + FAISS dense (`k=3`), combined via `EnsembleRetriever` (RRF) |
| Reranker | `flashrank` (`ms-marco-TinyBERT-L-2-v2`), reranks the ~6 hybrid candidates down to top-3 |
| Corrective retrieval | Grade → (generate \| rewrite query → retry retrieval, max 1 retry) → generate |
| Embedding model | `gemini-embedding-2-preview` |
| LLM | `gemini-2.5-flash`, `temperature=0.2`, `thinking_budget=0` |
| Conversation memory | LangGraph checkpointer, keyed by `thread_id` (SQLite dev / Postgres prod) |
| Deployment | FastAPI on Vercel (`api/server.py`) + Streamlit frontend (`front/streamlit.py`) |

**Measured quality — 50-question eval set, current config** (`ragas`; structure-aware loader + `chunk_size=350`, `chunk_overlap=100`): **faithfulness 0.952, answer relevancy 0.985, context recall 0.970, context precision 0.885.** All four metrics land within noise of the prior config (Run 2 below) — expected, since `app/loader.py`'s header/page-break fix only touches a handful of the 99 chunks, not enough to move a 50-question aggregate mean even when it fixes real, previously-broken questions. The evidence that matters is direct: the exact benchmark question "How many sick leave days are employees entitled to?" now scores precision/recall/faithfulness all ~1.0 with the correct "8 days" answer (previously the source of the "12/14 days" hallucinations), and a Laptop Policy Gate Pass sentence that used to be truncated at an old page break is now intact. See `RAG_EVALUATION_LOG_50Q.md` Run 5 for the full write-up, including a fourth, deeper PDF-extraction-order defect found (but not fixed) along the way.

**Measured quality — 50-question eval set, previous config** (`chunk_size=350`, `chunk_overlap=100`, page-level loader before the structure-aware fix): faithfulness 0.955, answer relevancy 0.970, context recall 0.990, context precision 0.882 — see `RAG_EVALUATION_LOG_50Q.md` Run 2 for the full write-up, including direct confirmation that a previously-wrong answer (a Laptop Policy question) became correct because a genuinely right chunk climbed into the retrieval top-3 (the step-size effect, distinct from the structure-aware fix above).

**Measured quality — 50-question eval set, previous config** (`chunk_size=300`, `chunk_overlap=75`, 2026-09-09, when the set grew from 40 to 50 with a 4th source document, `Laptop Policy Monit.pdf`, and 10 questions covering it): faithfulness 0.979, answer relevancy 0.962, context recall 0.880, context precision 0.838. Not directly comparable to the 40-question numbers below (larger, more topically diverse question set, different noise floor) — see `RAG_EVALUATION_LOG_50Q.md` Run 1 for the full breakdown, including two concretely-diagnosed retrieval misses on the new document (one a genuine "correct chunk ranked just outside the top-3" case, one a PDF-page-break chunking defect) and a per-policy score table.

**Measured quality — 40-question eval set (historical, for the version-history table below):** faithfulness 0.978, answer relevancy 0.979, context recall 0.950, context precision 0.877 (`chunk_overlap=75`). Within noise of the same config at `chunk_overlap=50` (faithfulness 0.958, relevancy 0.973, recall 0.975, precision 0.890) — expected, since the overlap increase mainly targets a specific chunk-boundary bug (see below), not the aggregate metrics. Corrective retrieval's rewrite path only triggered on 1 of 40 questions here, since this eval set contains no out-of-scope questions; its real value isn't visible in these numbers — manually asking something the HR docs don't cover (a dress-code question) confirmed the grader correctly flags insufficient context, retries with a rewritten query, and then honestly reports no information found instead of fabricating a policy, a safety property the current eval set can't score. See `RAG_EVALUATION_LOG_40Q.md` Run 21.

**Why `chunk_overlap` went 50 → 75 → 100 (and `chunk_size` 300 → 350):** manual testing found that two phrasings of the same sick-leave question gave contradictory answers ("14 days" vs. the correct "8 days" — verified directly against the source PDF, which never says "14 days of sick leave" anywhere). Root cause: at `chunk_size=300`, a chunk boundary landed mid-sentence exactly where a PDF page header interrupts a *different* leave type's "14 days" sentence (pilgrimage/annual leave), producing a garbled fragment that got conflated with genuine sick-leave content. Raising overlap to 75 avoided surfacing that specific garbled chunk in the top-3 for the two phrasings tested — **but the garbled fragment itself still exists verbatim in the corpus** (confirmed by direct inspection both at overlap=75 and again at the current 350/100), it just scores lower now. This remains a probabilistic mitigation, not a structural fix.

The `chunk_size=350`/`chunk_overlap=100` change (`RAG_EVALUATION_LOG_50Q.md` Run 2) is a related but distinct lever: it changes the *step size* between chunks (`chunk_size − chunk_overlap`, now 250, up from 225), producing fewer, less redundant chunks overall. This isn't about fixing broken chunks — it's about reducing how many near-duplicate chunks compete for a fixed `k=3` retrieval slot, which directly turned one previously-wrong answer correct (a genuinely right chunk climbed into the top-3 once there was less competition). Confirmed this doesn't fix the known structural defects — same two garbled/truncated fragments checked again at 350/100, still present unchanged at that point.

**The structural fix landed separately** (`RAG_EVALUATION_LOG_50Q.md` Run 5): `app/loader.py` now merges each PDF's pages into one continuous document before chunking (so a sentence spanning a page break is no longer guaranteed to split) and strips the repeated confidentiality-notice/"Issue #" boilerplate that used to land mid-sentence at page breaks. Directly re-verified both known defects are gone. This also surfaced a fourth, deeper defect — a `pypdf` text-extraction-order issue where a bulleted layout gets read out of visual order, actually dropping a sentence fragment rather than just interrupting it — which needs a layout-aware PDF parser to fix, not more chunking or regex work (`RAG_IMPROVEMENT_ROADMAP.md`).

## Version history & comparison

Every configuration below was measured with `ragas` (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) on the same 40-question HR-policy eval set, each score averaged across ≥1 independent run to separate real effects from run-to-run noise (~±0.03 per metric was the typical noise floor observed for a repeated run of the *same* config). The eval set grew to 50 questions after row 9 (see `RAG_EVALUATION_LOG_50Q.md`); rows below are historical and reflect the 40-question set they were actually measured on.

| # | Version | Faithfulness | Relevancy | Precision | Recall | Outcome |
|---|---|---|---|---|---|---|
| 1 | Baseline — plain FAISS, `k=3`, chunk 300/50 | 0.945 | 0.942 | 0.870 | 0.902 | Working baseline |
| 2 | Chunk size sweep — 200/50 and 400/50 | 0.897–0.946 | 0.905–0.962 | 0.850–0.881 | 0.819–0.869 | 300/50 stayed the best all-around; kept |
| 3 | Local cross-encoder reranker (retrieve `k=10` → rerank → top-3) | 0.958 | 0.956 | 0.869 | 0.888 | Improved quality slightly, but `sentence-transformers`/`torch` pushed the Vercel serverless function past its **500MB bundle limit**. Reverted. |
| 4 | Cohere Rerank API (HTTP-only, no local model) | — | — | — | — | Avoided the bundle-size problem, but the trial API key's **10 calls/minute** cap made it unusable under any real load (eval runs failed even with request pacing). Reverted. |
| 5 | Hybrid search — BM25 + FAISS via `EnsembleRetriever` (RRF), no reranker | **0.981** | **0.981** | 0.703 | **1.000** | Best and most consistent result across two independent runs — perfect recall, highest faithfulness/relevancy recorded. Real, repeatable trade-off: context_precision drops ~0.17 vs. plain retrieval, because the fused candidate set is larger and less filtered. Kept as the retrieval strategy, but not the final config — see #7. |
| 6 | + `thinking_budget=0`, conversation memory, cost monitoring | (same as #5) | | | | Latency/UX/observability additions layered on top of #5 without changing retrieval quality. |
| 7 | + `flashrank` reranker (`ms-marco-TinyBERT-L-2-v2`) on top of the hybrid candidates, kept top-3 | 0.971 | 0.975 | **0.890** | 0.975 | Two earlier reranker attempts (row 3, and a Cohere API attempt) were ruled out on infrastructure, not quality — `flashrank` avoids both problems (pure ONNX, ~235MB deploy-only footprint vs. Vercel's 500MB limit; runs locally, no rate limit). Closed most of the precision gap left by hybrid search (+0.187) at the cost of only small (~0.01-0.025) dips elsewhere. `top_k` swept to 2 (regression) and 4 (no change, extra tokens) — 3 confirmed as the local optimum. Kept as the retrieval strategy; superseded as "final" by row 8. |
| 8 | + corrective retrieval: `grade` node checks context sufficiency, `rewrite` + retry once if insufficient | 0.958 | 0.973 | 0.890 | 0.975 | The project's first agentic change — a decision node, not another retrieval/config tweak. Metrics are within noise of row 7 because the 40Q eval set contains no out-of-scope questions (the rewrite path fired on only 1/40). Real value confirmed manually instead: an out-of-scope question ("dress code," not in the docs) correctly triggered grade→rewrite→retry→honest "no information found" rather than a fabricated answer. Kept, but superseded as "final" by row 9. |
| 9 | + `chunk_overlap` 50 → 75 | 0.978 | 0.979 | 0.877 | 0.950 | Response to a live bug found during manual testing, not a config sweep: a chunk boundary at overlap=50 landed mid-sentence where a PDF page header interrupts a "14 days of pilgrimage/annual leave" sentence, and the garbled fragment got conflated with sick-leave content — one phrasing of a sick-leave question answered "14 days" (wrong; the source PDF says 8, with zero exceptions), another answered "8 days" (right). Corrective retrieval's grader didn't catch this since it checks sufficiency, not internal consistency. Raising overlap avoided retrieving that garbled chunk for the two phrasings tested, **but the same garbled fragment still exists verbatim in the corpus** — confirmed by direct inspection, it just scores lower now, so this is a mitigation, not a guaranteed fix. This also explains the precision/recall dip: smaller step size (300−75 vs 300−50) means more total chunks (104 vs 101) and more near-duplicate variants of messy boundary regions competing in the retrieval pool. Kept as the config — superseded as "final" by row 10, but "structure-aware chunking" (strip headers, split on sentence/page boundaries) remains the real open fix. |
| 10 | + 4th source document (`Laptop Policy Monit.pdf`) + 10 new eval questions — eval set grows from 40 to 50 | 0.979 | 0.962 | 0.838 | 0.880 | Not a config change — a corpus/eval-set growth milestone, moved to its own `RAG_EVALUATION_LOG_50Q.md` per this project's per-dataset-size logging convention. Precision/recall dropped vs. row 9, but not because the reranker or corrective retrieval regressed (the original 40 questions' scores held steady) — two concrete new chunking defects were found and diagnosed in the new document (both PDF page-break truncations, a different structural trigger than row 9's repeated-header bug), plus a new corpus-scaling concern: one correct chunk ranked #5/119 in BM25 and #8/119 in dense retrieval, just outside the fixed `k=3` cutoff — a near-miss that gets structurally more likely as more documents are added, independent of chunking quality. Kept, but superseded as "final" by row 11. |
| 11 | `chunk_size` 300 → 350, `chunk_overlap` 75 → 100 | 0.955 | 0.970 | **0.882** | **0.990** | Largest recall gain in either log (+0.110 vs. row 10), well beyond the noise floor, and confirmed as a real effect via direct chunk inspection rather than assumed from the metric alone: the previously-failing "who approves first" Laptop Policy question (row 10) is now answered correctly, because the mechanism here is *step size* (`chunk_size − chunk_overlap`, grown from 225 to 250) producing fewer, less redundant chunks — reducing competition for the fixed `k=3` retrieval slots let a genuinely correct chunk climb into the top-3. Re-checked all known structural chunking defects (row 9's header interruption, row 10's page-break truncations) at this new size/overlap: **all still present, byte-for-byte unchanged** — this lever reduces redundant competition, it doesn't fix broken chunks, so it's complementary to (not a substitute for) structure-aware chunking. Kept, but see row 12 for why "keep pushing the same direction" doesn't work. |
| 12 | `chunk_overlap` 100 → 50 (`chunk_size=350` unchanged) | 0.954 | 0.976 | 0.882 | 0.930 | Tested whether pushing the row-11 direction further (bigger step size again: 250 → 300) would help more — it didn't. Precision came back **bit-identical** to row 11 rather than improving, and recall *dropped* -0.06. More tellingly: the exact question row 11 fixed (**"who approves first"**) **regressed back to the wrong answer**, proving step size isn't a smooth dial — each distinct chunk_size/overlap pair produces entirely different chunk boundaries throughout the document, and whether a given fact's chunk ranks in the top-3 depends on those specific boundaries, not on a general "more/less redundancy" trend. **Reverted to row 11's config** — no reason to give back recall for zero precision gain. See `RAG_EVALUATION_LOG_50Q.md` Run 3. Three data points (rows 9, 11, 12) now show chunk-size/overlap tuning is a non-monotonic search space — further hand-tuning by intuition isn't reliable without the same full-eval-plus-inspection treatment each of these three got. |
| 13 | Tool-calling retrieval agent (LangGraph `ToolNode`, `.bind_tools()`, up to 4 self-directed searches) replacing the fixed corrective-retrieval graph | 0.918 | 0.897 | 0.808 | 0.853 | Rejected — every metric regressed well beyond noise vs. row 11/12, recall's **-0.137** the largest single-metric regression recorded in either eval log. Root cause, confirmed directly: the agent rephrases its own search query even for simple, already-clear questions (e.g. "sick leave entitlement" instead of the literal question), and that rephrasing genuinely retrieves worse chunks — one reproduced case had the actual number truncated in all 3 top chunks, and the model fabricated "12 days" (correct: 8) from nothing. The multi-search capability this was meant to unlock also didn't reliably trigger on a question designed to need it. **Reverted** — `main.py` is back to the corrective-retrieval graph (diff-verified identical to row 11/12). See `RAG_EVALUATION_LOG_50Q.md` Run 4. |
| 14 | Structure-aware loader: merge per-page `Document`s into one continuous document per file, strip repeated header/footer boilerplate, add paragraph breaks before bullets | 0.952 | 0.985 | 0.885 | 0.970 | Fixes the two structural chunking defects (header-interruption from row 9, page-break truncation from row 10) at the root instead of mitigating them via chunk-size tuning — `PyPDFLoader` returns one `Document` per page and `split_documents()` never merges across page boundaries, so a sentence spanning a page break was *guaranteed* to split, not just unlucky. All 4 metrics land within noise of row 11/12 (expected — the fix only touches a handful of the 99 chunks, not enough to move a 50-question aggregate), but direct verification confirms real fixes: the exact benchmark sick-leave question now scores ~1.0 on precision/recall/faithfulness, and the Laptop Policy Gate Pass sentence is no longer truncated. Also surfaced a **fourth, deeper defect** along the way (not fixed): a `pypdf` text-extraction-order issue where bulleted content gets read out of visual order, actually *dropping* a sentence fragment rather than just interrupting it — needs a layout-aware parser, not more chunking/regex work. **Kept — this is the current shipped config.** See `RAG_EVALUATION_LOG_50Q.md` Run 5. |

**Why hybrid search + reranker, rather than either alone:** hybrid search alone maximizes recall (perfect 1.000) but retrieves a larger, less-filtered candidate set, capping context_precision around 0.70. Two prior reranker-only attempts (row 3, and a Cohere API attempt) scored well but were ruled out on Vercel bundle size and rate limits respectively — infrastructure problems, not quality ones. Layering a lightweight reranker (`flashrank`) on top of hybrid's candidate set gets the best of both: hybrid still supplies a wide, high-recall candidate pool, and the reranker's joint query+passage relevance scoring (something neither RRF's rank-based fusion nor a plain similarity threshold can do — both were tried and failed, see `RAG_EVALUATION_LOG_40Q.md` Run 17) picks the genuinely most relevant 3 out of that pool.

**Why the small remaining trade-off is acceptable:** for an HR-policy assistant, a confidently wrong or incomplete answer (low faithfulness/recall) is a worse failure than a slightly noisier retrieved-context set (lower precision). The current config's small dips in faithfulness/relevancy/recall (-0.01 to -0.025) are within the ~0.03-0.05 noise floor observed throughout this project's evaluation history, while the precision gain is roughly 6x that noise floor — a good trade by the same reasoning used to accept hybrid search's original, larger precision cost.

## Project structure

```
P3/
├── main.py                    # LangGraph pipeline: retrieve → rerank → grade → (generate | rewrite → retry) → checkpoint → track usage
├── app/
│   ├── loader.py               # Loads PDFs from data/
│   ├── splitter.py             # Chunks documents
│   ├── vector_store.py         # Builds/loads the hybrid (BM25 + FAISS) retriever
│   ├── reranker.py              # flashrank reranker: hybrid candidates -> top-3 by relevance
│   ├── monitoring.py           # TokenCostEstimator: per-call token/cost accounting
│   └── usage_store.py          # Persists token usage to Postgres/SQLite, serves aggregate totals
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

Runs every question in `rag_eval_dataset.json` (50 as of 2026-09-09) through the pipeline (each on a fresh `thread_id`, so eval questions don't share conversation memory with each other) and scores the results with `ragas`. Detailed per-run history and reasoning behind every configuration change live in `RAG_EVALUATION_LOG.md` / `RAG_EVALUATION_LOG_40Q.md` / `RAG_EVALUATION_LOG_50Q.md` (kept locally, not committed — see the version comparison above for the summarized findings).

## Adding documents

Drop additional PDFs into `data/`, then delete the `vector_store/` directory so the index gets rebuilt on the next run.
