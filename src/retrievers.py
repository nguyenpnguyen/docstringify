import logging

from langchain_core.documents import Document
from src.db import CodeChunk, select_code_chunk_by_name, get_dependencies


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def chunk_to_document(chunk: CodeChunk) -> Document:
    return Document(
        page_content=chunk.code,
        metadata={
            "path": chunk.path,
            "name": chunk.name,
            "type": chunk.type,
            "parent_class": chunk.parent_class,
            "line_number": chunk.line_number,
            "docstring": chunk.docstring,
        },
    )

def dependency_retriever(name: str, path: str) -> list[Document]:
    chunk = select_code_chunk_by_name(name, path)
    return get_dependencies(chunk)


def usage_retriever(chunk: CodeChunk) -> list[Document]:
    pass

def retrieve_relevant_docs(code: str) -> list[Document]:
    pass
