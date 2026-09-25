import json
import sys
import types
import uuid

import pandas as pd

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
from ragas.run_config import RunConfig
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
    # Each eval question is independent - use a fresh thread_id per call so the
    # checkpointer doesn't carry unrelated questions' history into this answer.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    current_state = {"question": question, "documents": [], "answer": ""}
    current_state = compiled_graph.invoke(current_state, config=config)
    return {
        "answer": current_state["answer"],
        "contexts": [d.page_content for d in current_state["documents"]]
    }


# Out-of-scope (irrelevant) questions have no real ground-truth context to
# compare retrieval against, so context_precision/context_recall would score
# near-zero by construction (there's no chunk that "supports" a reference
# saying the docs don't cover this) - that's expected, correct behavior, not
# a retrieval failure. They're scored separately below via abstention
# detection instead of being run through the standard ragas metrics, which
# stay comparable to the 40Q/50Q logs (in-scope questions only).
in_scope_items = [item for item in eval_data if item.get("scope", "in_scope") == "in_scope"]
out_of_scope_items = [item for item in eval_data if item.get("scope") == "out_of_scope"]

ABSTENTION_PHRASES = [
    "don't have", "do not have", "doesn't have", "does not have",
    "don't know", "do not know",
    "no information", "not have information", "doesn't contain", "do not contain",
    "does not contain", "not covered", "doesn't cover", "does not cover",
    "not mentioned", "not addressed", "outside the scope", "out of scope",
    "cannot find", "can't find", "unable to find", "i'm not able to find",
    "not available in", "not provided in", "no mention of",
]


def looks_like_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


results = []
for item in in_scope_items:
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
    run_config=RunConfig(max_workers=4, timeout=300),
)

print(f"\n===== RAG EVALUATION (in-scope, n={len(in_scope_items)}) =====")
print(evaluation)

evaluation.to_pandas().to_csv(
    "ragas_results.csv",
    index=False,
)

print("\nSaved: ragas_results.csv")

abstention_results = []
correct_abstentions = 0
for item in out_of_scope_items:
    result = run_rag(item["user_input"])
    abstained = looks_like_abstention(result["answer"])
    correct_abstentions += abstained
    abstention_results.append(
        {
            "user_input": item["user_input"],
            "response": result["answer"],
            "abstained": abstained,
        }
    )

print(f"\n===== ABSTENTION CHECK (out-of-scope, n={len(out_of_scope_items)}) =====")
print(f"Correctly abstained: {correct_abstentions}/{len(out_of_scope_items)}")

pd.DataFrame(abstention_results).to_csv("ragas_abstention_results.csv", index=False)
print("Saved: ragas_abstention_results.csv (review manually - keyword match is a heuristic, not ground truth)")
