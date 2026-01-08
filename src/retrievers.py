import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def retrieve_relevant_docs(query: str, vector_store: VectorStore, k: int = 4) -> List[Document]:
    """
    Useful for finding code snippets, function definitions, or examples
    semantically related to a specific topic or function name.
    """
    return vector_store.similarity_search(query=query, k=k)
