import pytest
from langchain_core.documents import Document

from src.db import build_call_graph, get_or_create_code_chunk

# A new document for a more complex test case for retrieve_relevant_docs
DOC_D = Document(
    page_content="def func_d(self):\n    self.func_a()",
    metadata={
        "path": "test.py",
        "name": "func_d",
        "type": "method",
        "parent_class": "MyClass",
        "line_number": 10,
        "calls": ["func_a"],
    },
)

@pytest.fixture
def extended_populated_db(populated_db):
    """
    Extends the populated_db with an additional code chunk for more complex scenarios.
    """
    get_or_create_code_chunk(DOC_D)
    return populated_db


@pytest.fixture(autouse=True)
def setup_call_graph(extended_populated_db):
    """
    Ensures the call graph is built for tests that rely on dependencies or dependents.
    This fixture will run automatically for every test in this file.
    """
    build_call_graph()


def test_dependency_retriever_finds_dependencies():
    """
    Tests that the dependency_retriever correctly returns documents
    that are called by the specified code chunk.
    """
    from src.retrievers import dependency_retriever

    # In conftest.py, func_a is defined as calling func_b.
    dependencies = dependency_retriever("def func_a(self):\n    \"\"\"Doc A\"\"\"\n    self.func_b()")

    assert isinstance(dependencies, list)
    assert len(dependencies) == 1
    assert isinstance(dependencies[0], Document)
    assert dependencies[0].metadata["name"] == "func_b"


def test_usage_retriever_finds_dependents():
    """
    Tests that the usage_retriever correctly returns documents
    that call the specified code chunk.
    """
    from src.retrievers import usage_retriever

    # func_a is called by func_d
    dependents = usage_retriever("def func_a(self):\n    \"\"\"Doc A\"\"\"\n    self.func_b()")

    assert isinstance(dependents, list)
    assert len(dependents) == 1
    assert isinstance(dependents[0], Document)
    assert dependents[0].metadata["name"] == "func_d"


def test_retrieve_relevant_docs_combines_dependencies_and_usage():
    """
    Tests that retrieve_relevant_docs returns a combined and de-duplicated
    list of both dependencies (callees) and dependents (callers).
    """
    from src.retrievers import retrieve_relevant_docs

    # For 'func_a':
    # - Dependencies: [func_b]
    # - Dependents: [func_d]
    # Expected result is a list containing documents for func_b and func_d.
    results = retrieve_relevant_docs("func_a")

    assert isinstance(results, list)
    assert len(results) == 2

    result_names = {doc.metadata["name"] for doc in results}
    assert result_names == {"func_b", "func_d"}


def test_retrieve_relevant_docs_handles_one_way_relations():
    """
    Tests retrieve_relevant_docs for a chunk that has dependents but no dependencies.
    """
    from src.retrievers import retrieve_relevant_docs
    
    # For 'func_b':
    # - Dependencies: []
    # - Dependents: [func_a]
    results = retrieve_relevant_docs("func_b")
    assert len(results) == 1
    assert results[0].metadata["name"] == "func_a"

def test_retrieve_relevant_docs_handles_no_relations():
    """
    Tests retrieve_relevant_docs for a chunk with no connections.
    """
    from src.retrievers import retrieve_relevant_docs
    
    # func_c has no dependencies and no dependents in the test data
    results = retrieve_relevant_docs("func_c")
    assert len(results) == 0

def test_retrieve_relevant_docs_deduplicates_results():
    """
    Tests that if a function is both a dependency and a dependent (circular),
    it appears only once in the final list.
    """
    # This requires a new test setup for a circular dependency.
    # We will mock the underlying retrievers for simplicity.
    from src.retrievers import retrieve_relevant_docs
    from unittest.mock import patch

    doc_a = Document(page_content="def func_a(): pass", metadata={"name": "func_a"})
    doc_b = Document(page_content="def func_b(): pass", metadata={"name": "func_b"})

    with patch('src.retrievers.dependency_retriever') as mock_dep, \
         patch('src.retrievers.usage_retriever') as mock_usage:
        
        # Simulate func_x calling func_a, and being called by func_b and func_a
        mock_dep.return_value = [doc_a]
        mock_usage.return_value = [doc_a, doc_b]

        results = retrieve_relevant_docs("func_x")
        
        # Should contain doc_a and doc_b, with doc_a appearing only once.
        assert len(results) == 2
        result_names = {doc.metadata["name"] for doc in results}
        assert result_names == {"func_a", "func_b"}

