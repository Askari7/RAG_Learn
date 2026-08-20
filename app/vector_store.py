from app.splitter import split_documents
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def create_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    index_path = "vector_store/index.faiss"
    pkl_path = "vector_store/index.pkl"

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
        splits = split_documents()

        vector_store = FAISS.from_documents(
            splits,
            embeddings
        )

        vector_store.save_local("vector_store")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    return vector_store, retriever