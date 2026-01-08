import pytest
from langchain_core.documents import Document
from src.injector import index_codebase
from src.loader import load_embeddings, load_vector_store

# Sample documents to be indexed in the test vector store
TEST_DOCS = [
    Document(
        page_content="def add(a: int, b: int) -> int:\n    \"\"\"Adds two integers.\"\"\"\n    return a + b",
        metadata={"source": "math_utils.py"}
    ),
    Document(
        page_content="def subtract(a: int, b: int) -> int:\n    \"\"\"Subtracts second integer from the first.\"\"\"\n    return a - b",
        metadata={"source": "math_utils.py"}
    ),
]

@pytest.fixture(scope="session")
def test_vector_store():
    """
    Pytest fixture to create and populate a test vector store.
    This fixture has a 'session' scope, so it's created once per test session.
    """
    # Use the real embedding model from the application for a more realistic test
    # This assumes an Ollama instance is running, as per the project setup.
    embedding_model = load_embeddings(model="qwen3-embedding:0.6b")

    # Use an in-memory Chroma instance for testing to avoid creating files
    vector_store = load_vector_store(
        collection_name="test_collection",
        embedding=embedding_model,
    )

    # Index the sample documents
    uuids = [str(i) for i in range(len(TEST_DOCS))]
    vector_store.add_documents(documents=TEST_DOCS, ids=uuids)

    return vector_store
