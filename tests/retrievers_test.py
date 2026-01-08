from langchain_core.vectorstores import VectorStore
from src.retrievers import retrieve_relevant_docs


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
