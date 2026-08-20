from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    question: str,
    documents: list[Document],
    top_k: int = 2,
) -> list[Document]:

    if not documents:
        return []

    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        document
        for document, score in ranked[:top_k]
    ]