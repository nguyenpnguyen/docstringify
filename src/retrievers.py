from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Constants for persistence paths
DB_DIR = "./chroma_db"

def index_codebase(documents: List[Document], embedding_model):
    """
    Takes a list of Documents (from src/ingest.py) and persists them to ChromaDB.
    """
    if not documents:
        print("No documents to index.")
        return

    print(f"Indexing {len(documents)} chunks to ChromaDB...")
    
    # Store in ChromaDB (Semantic)
    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=DB_DIR,
        collection_name="docstringify_code"
    )
    print("Indexing Complete.")

def get_semantic_retriever(embedding_model, k_semantic=4):
    """
    Returns a naive semantic retriever using only Vector Search (ChromaDB).
    This serves as the baseline for performance comparison.
    """
    print("Initializing Naive Semantic Retriever...")
    
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_name="docstringify_code"
    )
    
    # We return the standard LangChain retriever interface
    return vectorstore.as_retriever(search_kwargs={"k": k_semantic})
