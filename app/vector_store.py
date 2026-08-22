from app.splitter import split_documents
import os
import pickle
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
    bm25_path = "vector_store/bm25.pkl"

    faiss_cached = os.path.exists(index_path) and os.path.exists(pkl_path)
    bm25_cached = os.path.exists(bm25_path)

    # Only need the source PDFs (data/) when a cache is missing and has to be
    # rebuilt. Once both caches exist, neither retriever touches data/ again.
    splits = None if (faiss_cached and bm25_cached) else split_documents()

    # Load existing vector store
    if faiss_cached:
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

    # Load existing BM25 index
    if bm25_cached:
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)

    # Build and cache the BM25 index if it doesn't exist
    else:
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3

        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5],
    )

    return vector_store, retriever