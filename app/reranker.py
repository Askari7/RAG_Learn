from typing import List
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

# TinyBERT-L-2: smallest flashrank model (~4MB, ONNX, no torch) - the whole
# point of retrying reranking is staying well under Vercel's 500MB bundle
# limit, which killed the sentence-transformers/torch cross-encoder before.
_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")


def rerank(query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
    if not documents:
        return documents

    passages = [
        {"id": i, "text": doc.page_content} for i, doc in enumerate(documents)
    ]
    ranked = _ranker.rerank(RerankRequest(query=query, passages=passages))
    top_ids = [r["id"] for r in ranked[:top_k]]
    return [documents[i] for i in top_ids]
