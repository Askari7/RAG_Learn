from app.splitter import split_documents
import os
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def create_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    index_path = "vector_store/index.faiss"
    pkl_path = "vector_store/index.pkl"

    # BM25 has no on-disk index to load - it needs the raw splits every time.
    splits = split_documents()

    # Load existing vector store
    if os.path.exists(index_path) and os.path.exists(pkl_path):
        vector_store = FAISS.load_local(
            "vector_store",
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )

    # Create vector store if it doesn't exist
    else:
        vector_store = FAISS.from_documents(
            splits,
            embeddings
        )

        vector_store.save_local("vector_store")

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5],
    )

    return vector_store, retriever