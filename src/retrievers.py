from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

def get_semantic_retriever(vector_store: VectorStore):
    """Strategy A: Pure Semantic Search using ChromaDB."""
    return vector_store.as_retriever(search_kwargs={"k": 4})

def get_bm25_retriever(documents: Document):
    """Strategy B: BM25 Keyword Search."""
    return BM25Retriever.from_documents([documents])
