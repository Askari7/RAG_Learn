# P3 — HR Policy RAG Assistant

A small Retrieval-Augmented Generation (RAG) pipeline that answers questions about a set of HR policy PDFs using Google Gemini and a local FAISS vector store, orchestrated with LangGraph.

## How it works

1. **Load** — `app/loader.py` reads every PDF under `data/` (`DirectoryLoader` + `PyPDFLoader`).
2. **Split** — `app/splitter.py` chunks the loaded documents (`RecursiveCharacterTextSplitter`, 1000 chars, 50 overlap).
3. **Embed & Index** — `app/vector_store.py` embeds the chunks with `GoogleGenerativeAIEmbeddings` (`models/text-embedding-004`) and stores them in a FAISS index under `vector_store/`. On subsequent runs it loads the cached index instead of re-embedding.
4. **Retrieve & Answer** — `main.py` builds a one-node LangGraph graph: given a question, it retrieves the top-3 most similar chunks, builds a context string, and asks `gemini-2.5-flash` to answer grounded in that context.

## Project structure

```
P3/
├── main.py                # Entry point: builds and runs the LangGraph pipeline
├── app/
│   ├── loader.py           # Loads PDFs from data/
│   ├── splitter.py         # Splits documents into chunks
│   └── vector_store.py     # Builds/loads the FAISS index and retriever
├── data/                   # Source HR policy PDFs
├── vector_store/           # Cached FAISS index (index.faiss / index.pkl)
└── pyproject.toml
```

## Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Google Gemini API key

## Setup

```bash
uv sync
```

Create a `.env` file in the project root with:

```
GEMINI_API_KEY=your-gemini-api-key

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=your-project-name
```

## Usage

```bash
uv run main.py
```

This runs a sample question ("What is the main topic of the documents?") against the indexed policies and prints the answer. Edit the `question` passed to `compiled_graph.invoke(...)` in `main.py` to ask something else.

## Adding documents

Drop additional PDFs into `data/`, then delete the `vector_store/` directory so the index gets rebuilt on the next run.
