from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated, List, TypedDict
from langchain_core.documents  import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from app.metadata_filter import infer_source
from app.monitoring import TokenCostEstimator
from app.reranker import rerank
from app.telemetry import get_tracer
from app.usage_store import init_usage_table, record_usage
from app.vector_store import create_vector_store
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
import os
import sqlite3
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

DB_URI = os.getenv("DATABASE_URL")

if DB_URI:
    pg_conn = psycopg.connect(DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    checkpointer = PostgresSaver(pg_conn)
else:
    sqlite_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(sqlite_conn)

checkpointer.setup()

init_usage_table()

# Corrective RAG: if the grader rejects the retrieved context, rewrite the query
# and retry retrieval once before giving up and answering with what's available.
MAX_CORRECTIVE_RETRIES = 1

class MessagesState(TypedDict):
    question: str
    search_query: str
    documents: List[Document]
    context_sufficient: bool
    retry_count: int
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]

vector_store, retriever, bm25_by_source = create_vector_store()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, thinking_budget=0)

# gemini-2.5-flash standard (non-batch) text pricing: $0.30 / 1M input tokens, $2.50 / 1M output tokens
token_cost_estimator = TokenCostEstimator(input_cost_per_1m=0.30, output_cost_per_1m=2.50)

tracer = get_tracer()


def _track_usage(llm_response):
    usage = llm_response.usage_metadata
    if usage:
        token_cost_estimator.add_usage(usage["input_tokens"], usage["output_tokens"])
        call_cost = (
            usage["input_tokens"] / 1_000_000 * token_cost_estimator.input_cost_per_1m
            + usage["output_tokens"] / 1_000_000 * token_cost_estimator.output_cost_per_1m
        )
        record_usage(usage["input_tokens"], usage["output_tokens"], call_cost)


def _retriever_for(query: str):
    # Only narrow the candidate pool when the query confidently matches one
    # policy and no others; otherwise fall back to the full, unfiltered
    # retriever (see app/metadata_filter.py for why).
    source = infer_source(query)
    if source is None or source not in bm25_by_source:
        return retriever

    scoped_dense = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": {"source": source}},
    )
    return EnsembleRetriever(
        retrievers=[bm25_by_source[source], scoped_dense],
        weights=[0.5, 0.5],
    )


def retrieve_node(state: MessagesState):
    with tracer.start_as_current_span("retrieve") as span:
        query = state.get("search_query") or state["question"]
        source = infer_source(query)
        results = rerank(query, _retriever_for(query).invoke(query), top_k=3)
        span.set_attribute("search_query", query)
        span.set_attribute("documents.count", len(results))
        span.set_attribute("metadata_filter.source", source or "none")
        return {"documents": results}


def grade_node(state: MessagesState):
    with tracer.start_as_current_span("grade") as span:
        question = state["question"]
        context = "\n\n".join(d.page_content for d in state["documents"])
        prompt = (
            f"Question: {question}\n\nRetrieved context:\n{context}\n\n"
            "Does this context contain enough information to give a complete, accurate answer "
            "to the question? Reply with exactly one word: YES or NO."
        )
        llm_response = llm.invoke(prompt)
        _track_usage(llm_response)
        context_sufficient = llm_response.content.strip().upper().startswith("YES")
        span.set_attribute("context_sufficient", context_sufficient)
        return {"context_sufficient": context_sufficient}


def route_after_grade(state: MessagesState):
    if state["context_sufficient"] or state.get("retry_count", 0) >= MAX_CORRECTIVE_RETRIES:
        return "generate"
    return "rewrite"


def rewrite_node(state: MessagesState):
    with tracer.start_as_current_span("rewrite") as span:
        question = state["question"]
        context = "\n\n".join(d.page_content for d in state["documents"])
        prompt = (
            f"Original question: {question}\n\n"
            f"This search retrieved context judged insufficient to answer it:\n{context}\n\n"
            "Rewrite the question as a better, more specific search query to find the missing "
            "information. Reply with only the rewritten query, no explanation."
        )
        llm_response = llm.invoke(prompt)
        _track_usage(llm_response)
        retry_count = state.get("retry_count", 0) + 1
        span.set_attribute("retry_count", retry_count)
        return {
            "search_query": llm_response.content.strip(),
            "retry_count": retry_count,
        }


def generate_node(state: MessagesState):
    with tracer.start_as_current_span("generate") as span:
        question = state["question"]
        history = state.get("messages", [])
        context = "\n\n".join(d.page_content for d in state["documents"])
        prompt = history + [HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")]
        llm_response = llm.invoke(prompt)
        _track_usage(llm_response)
        usage = llm_response.usage_metadata
        if usage:
            span.set_attribute("llm.input_tokens", usage["input_tokens"])
            span.set_attribute("llm.output_tokens", usage["output_tokens"])
        return {
            "answer": llm_response.content,
            "messages": [HumanMessage(content=question), AIMessage(content=llm_response.content)],
        }


graph = StateGraph(MessagesState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate", END)

compiled_graph = graph.compile(checkpointer=checkpointer)
