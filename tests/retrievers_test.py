import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import VectorStore
from src.retrievers import index_codebase, retrieve_relevant_docs
from src.loader import load_vector_store


@pytest.fixture
def mock_embedding_model() -> FakeEmbeddings:
    """Fixture for a fake embedding model."""
    return FakeEmbeddings(size=384)


@pytest.fixture
def in_memory_vector_store(mock_embedding_model: FakeEmbeddings) -> VectorStore:
    """Fixture for a clean, in-memory vector store for each test."""
    return load_vector_store(
        collection_name="test_retriever_collection",
        embedding=mock_embedding_model,
    )


def test_index_codebase(in_memory_vector_store: VectorStore, mock_embedding_model: FakeEmbeddings):
    """
    Unit test for the index_codebase function.
    """
    test_docs = [
        Document(page_content="def test_func(): pass", metadata={"source": "test.py"})
    ]

    # Pass the in-memory vector store to the function
    vs = index_codebase(
        documents=test_docs,
        embedding_model=mock_embedding_model,
        vector_store=in_memory_vector_store
    )

    # 1. Check if the function returns the correct object
    assert vs == in_memory_vector_store, "Function should return the vector store instance."

    # 2. Check if the document was actually added by performing a search
    results = vs.similarity_search("test_func", k=1)
    assert len(results) > 0, "No documents were found after indexing."
    assert results[0].page_content == "def test_func(): pass"


def test_retrieve_relevant_docs(test_vector_store: VectorStore):
    """
    Unit test for the retrieve_relevant_docs function.
    This test uses the pre-populated vector store from conftest.py.
    """
    # The query is semantically similar to "def add(a: int, b: int)..."
    query = "function for adding two numbers"

    retrieved_docs = retrieve_relevant_docs(query=query, vector_store=test_vector_store, k=1)

    # Assert that we got some documents back
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) == 1, "Expected to retrieve one document."

    # Assert that the content of the most relevant document is what we expect
    assert "def add(a: int, b: int)" in retrieved_docs[0].page_content