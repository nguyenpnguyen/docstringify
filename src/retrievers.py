import logging

from langchain_core.documents import Document
from src.db import CodeChunk


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def dependency_retriever(chunk: CodeChunk) -> list[Document]:
    pass

def usage_retriever(chunk: CodeChunk) -> list[Document]:
    pass

def retrieve_relevant_docs(code: str) -> list[Document]:
    pass
