from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated, List, TypedDict
from langchain_core.documents  import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from app.monitoring import TokenCostEstimator
from app.usage_store import init_usage_table, record_usage
from app.vector_store import create_vector_store
from dotenv import load_dotenv
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
class MessagesState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]

vector_store, retriever = create_vector_store()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, thinking_budget=0)

# gemini-2.5-flash standard (non-batch) text pricing: $0.30 / 1M input tokens, $2.50 / 1M output tokens
token_cost_estimator = TokenCostEstimator(input_cost_per_1m=0.30, output_cost_per_1m=2.50)

def llm_node(state: MessagesState):
    question = state["question"]
    history = state.get("messages", [])
    results = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in results)
    prompt = history + [HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")]
    llm_response = llm.invoke(prompt)
    usage = llm_response.usage_metadata
    if usage:
        token_cost_estimator.add_usage(usage["input_tokens"], usage["output_tokens"])
        call_cost = (
            usage["input_tokens"] / 1_000_000 * token_cost_estimator.input_cost_per_1m
            + usage["output_tokens"] / 1_000_000 * token_cost_estimator.output_cost_per_1m
        )
        record_usage(usage["input_tokens"], usage["output_tokens"], call_cost)
    return {
        "answer": llm_response.content,
        "documents": results,
        "messages": [HumanMessage(content=question), AIMessage(content=llm_response.content)],
    }

graph = StateGraph(MessagesState)
graph.add_node("llm_node", llm_node)
graph.add_edge(START, "llm_node")
graph.add_edge("llm_node", END)

compiled_graph = graph.compile(checkpointer=checkpointer)