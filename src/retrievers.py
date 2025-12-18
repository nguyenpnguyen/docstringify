from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever


def index_codebase(documents: List[Document], embedding_model: Embeddings) -> VectorStore:
    """
    Takes a list of Documents and persists.
    """
    pass


def _get_semantic_retriever(embedding_model: Embeddings, k_semantic=4) -> BaseRetriever:
    """
    Returns a naive semantic retriever using only Vector Search.
    """
    pass


@tool
def retrieve_context(retriever: BaseRetriever, query: str) -> List[Document]:
    """
    Retrieves relevant code snippets given a query.
    """
    pass
