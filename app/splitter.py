from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.loader import loading_documents

def split_documents():
    documents = loading_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100)
    splits = splitter.split_documents(documents)
    return splits
