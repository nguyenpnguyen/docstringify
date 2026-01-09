import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def dependency_retriever():
    pass

def usage_retriever():
    pass

def recipropal_rank_fusion():
    pass

def retrieve_relevant_docs(query: str, vector_store: VectorStore, k: int = 4) -> list[Document]:
    pass
