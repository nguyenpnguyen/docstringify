import os
from typing import List

from src.loader import load_vector_store
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from uuid import uuid4


def index_codebase(documents: List[Document], embedding_model: Embeddings) -> VectorStore:
    """
    Takes a list of Documents and persists.
    """
    # find current directory name (only name not absolute path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(current_dir)

    vector_store = load_vector_store(collection_name=project_name, embedding=embedding_model)

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    return vector_store


def _get_semantic_retriever(embedding_model: Embeddings, k_semantic=4) -> BaseRetriever:
    """
    Returns a naive semantic retriever using only Vector Search.
    """
    pass


def retrieve_relevant_docs(query: str, vector_store: VectorStore, k: int = 4) -> List[Document]:
    """
    Useful for finding code snippets, function definitions, or examples
    semantically related to a specific topic or function name.
    """
    return vector_store.similarity_search(query=query, k=k)
