import logging

from langchain_core.documents import Document


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def dependency_retriever():
    pass

def usage_retriever():
    pass

def retrieve_relevant_docs(query: str, k: int = 10) -> list[Document]:
    pass
