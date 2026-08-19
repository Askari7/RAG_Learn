from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, TypedDict
from langchain_core.documents  import Document
from langgraph.graph import StateGraph, START, END
from app.vector_store import create_vector_store
from dotenv import load_dotenv
load_dotenv()
class MessagesState(TypedDict):
    question: str
    documents: List[Document]
    answer: str

vector_store, retriever = create_vector_store()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def llm_node(state: MessagesState):
    question = state["question"]    
    results = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in results)
    llm_response = llm.invoke(f"Context:\n{context}\n\nQuestion: {question}")
    return {"answer": llm_response.content, "documents": results}

graph = StateGraph(MessagesState)
graph.add_node("llm_node", llm_node)
graph.add_edge(START, "llm_node")
graph.add_edge("llm_node", END)

compiled_graph = graph.compile()

response = compiled_graph.invoke({"question": "Briefly, explain consequences of leave?"})
print("Answer:", response["answer"])