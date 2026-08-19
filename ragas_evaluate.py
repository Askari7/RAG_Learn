import json
import sys
import types

# ragas 0.4.3 hard-imports ChatVertexAI from a module path that langchain-community
# removed in 0.4.x (github.com/explodinggradients/ragas/issues/2753). We don't use
# Vertex AI, so stub it out instead of pinning an older langchain-community.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from main import compiled_graph, llm

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
)
with open("rag_eval_dataset.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)


def run_rag(question: str):
    current_state = {"question": question, "documents": [], "answer": ""}
    current_state = compiled_graph.invoke(current_state)
    return {
        "answer": current_state["answer"],
        "contexts": [d.page_content for d in current_state["documents"]]
    }


results = []

for item in eval_data:
    result = run_rag(item["user_input"])

    results.append(
        {
            "user_input": item["user_input"],
            "reference": item["reference"],
            "response": result["answer"],
            "retrieved_contexts": result["contexts"],
        }
    )

dataset = Dataset.from_list(results)

evaluation = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print("\n===== RAG EVALUATION =====")
print(evaluation)

evaluation.to_pandas().to_csv(
    "ragas_results.csv",
    index=False,
)

print("\nSaved: ragas_results.csv")
