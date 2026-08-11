from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
def loading_documents():
    loader = DirectoryLoader(
        "./data", glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents
