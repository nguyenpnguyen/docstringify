import os
from typing import List

from langchain_community.vectorstores.sqlitevec import SQLiteVec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from src.code_parser import CodeChunkMetadata


def index_codebase(documents: List[Document], embedding_model: Embeddings) -> VectorStore:
    """
    Takes a list of Documents and persists.
    """
    # find current directory name (only name not absolute path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(current_dir)
    db_file = os.path.join(current_dir, f"data/{project_name}_vec.db")

    conn = SQLiteVec.create_connection(db_file=db_file)

    if os.path.exists(db_file):
        os.remove(db_file)

    vector_store = SQLiteVec.from_documents(
        documents,
        embedding_model,
        table=f"{project_name}_table",
        connection=conn,
        db_file=db_file
    )

    content: List[str] = [doc.page_content for doc in documents if doc.page_content.strip()]
    metadata_text: List[dict] = [doc.metadata for doc in documents if doc.metadata.strip()]

    metadata_lst: List[CodeChunkMetadata] = [
        CodeChunkMetadata.model_validate(meta) for meta in metadata_text
    ]

    vector_store.add_texts(texts=content, metadatas=metadata_text)

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
