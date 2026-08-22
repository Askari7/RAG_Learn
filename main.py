from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, TypedDict
from langchain_core.documents  import Document
from langgraph.graph import StateGraph, START, END
from app.monitoring import TokenCostEstimator
from app.usage_store import init_usage_table, record_usage
from app.vector_store import create_vector_store
from dotenv import load_dotenv
load_dotenv()

init_usage_table()
class MessagesState(TypedDict):
    question: str
    documents: List[Document]
    answer: str

vector_store, retriever = create_vector_store()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, thinking_budget=0)

# gemini-2.5-flash standard (non-batch) text pricing: $0.30 / 1M input tokens, $2.50 / 1M output tokens
token_cost_estimator = TokenCostEstimator(input_cost_per_1m=0.30, output_cost_per_1m=2.50)

def llm_node(state: MessagesState):
    question = state["question"]
    results = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in results)
    llm_response = llm.invoke(f"Context:\n{context}\n\nQuestion: {question}")
    usage = llm_response.usage_metadata
    if usage:
        token_cost_estimator.add_usage(usage["input_tokens"], usage["output_tokens"])
        call_cost = (
            usage["input_tokens"] / 1_000_000 * token_cost_estimator.input_cost_per_1m
            + usage["output_tokens"] / 1_000_000 * token_cost_estimator.output_cost_per_1m
        )
        record_usage(usage["input_tokens"], usage["output_tokens"], call_cost)
    return {"answer": llm_response.content, "documents": results}

graph = StateGraph(MessagesState)
graph.add_node("llm_node", llm_node)
graph.add_edge(START, "llm_node")
graph.add_edge("llm_node", END)

compiled_graph = graph.compile()