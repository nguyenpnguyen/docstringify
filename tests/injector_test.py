import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import VectorStore
from src.injector import index_codebase, load_and_split_repository
from src.loader import load_vector_store
from pathlib import Path
import os


@pytest.fixture
def mock_embedding_model() -> FakeEmbeddings:
    """Fixture for a fake embedding model."""
    return FakeEmbeddings(size=384)


@pytest.fixture
def in_memory_vector_store(mock_embedding_model: FakeEmbeddings) -> VectorStore:
    """Fixture for a clean, in-memory vector store for each test."""
    return load_vector_store(
        collection_name="test_injector_collection",
        embedding=mock_embedding_model,
    )


@pytest.fixture
def sample_repo(tmp_path):
    """Fixture to create a temporary sample repository for testing."""
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()

    # Create a simple Python file
    (repo_dir / "my_module.py").write_text("""
def func_a():
    \"\"\"A simple function.\"\"\"
    return 1

class MyClass:
    def method_b(self):
        \"\"\"A simple method.\"\"\"
        return 2
""")

    # Create another Python file
    (repo_dir / "another_module.py").write_text("""
def func_c():
    \"\"\"Another function.\"\"\"
    return 3
""")
    yield repo_dir
    # tmp_path is automatically cleaned up by pytest


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


def test_load_and_split_repository(sample_repo):
    """
    Integration test for load_and_split_repository.
    """
    docs = load_and_split_repository(str(sample_repo))

    assert len(docs) > 0, "No documents were created from the sample repository."

    # Expecting at least func_a, MyClass (with __init__ implicit), method_b, func_c
    # The splitting might create more, but at least these logical units should be there
    expected_names = {"func_a", "MyClass", "method_b", "func_c"}
    found_names = {doc.metadata["name"] for doc in docs if "name" in doc.metadata}

    assert expected_names.issubset(found_names), f"Missing expected code structures. Found: {found_names}"

    # Check for metadata
    for doc in docs:
        assert "source" in doc.metadata
        assert "name" in doc.metadata
        assert "type" in doc.metadata
        assert doc.metadata["source"].startswith(str(sample_repo))
        assert doc.page_content is not None
        assert len(doc.page_content) > 0

    # Ensure no empty documents (might happen with some splitters if not careful)
    assert all(doc.page_content.strip() != "" for doc in docs)
