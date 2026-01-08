import os
import logging
from typing import List

from src.loader import load_vector_store
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def index_codebase(documents: List[Document], embedding_model: Embeddings, vector_store: VectorStore) -> VectorStore:
    """
    Takes a list of Documents and persists them to a vector store.
    If no vector_store is provided, one will be created.
    """
    if vector_store is None:
        # find current directory name (only name not absolute path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_name = os.path.basename(current_dir)
        vector_store = load_vector_store(collection_name=project_name, embedding=embedding_model)

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    return vector_store


def retrieve_relevant_docs(query: str, vector_store: VectorStore, k: int = 4) -> List[Document]:
    """
    Useful for finding code snippets, function definitions, or examples
    semantically related to a specific topic or function name.
    """
    return vector_store.similarity_search(query=query, k=k)
