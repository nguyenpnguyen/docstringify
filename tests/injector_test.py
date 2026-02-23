import pytest
from docstringify.injector import index_documents, load_and_split_repository
from docstringify.db import CodeChunk, CallGraph, select_code_chunk_by_name, get_dependencies


@pytest.fixture
def sample_repo(tmp_path):
    """Fixture to create a temporary sample repository for testing."""
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()

    (repo_dir / "my_module.py").write_text("""
def func_a():
    \"\"\"A simple function that calls another.\"\"\"
    func_b()

def func_b():
    \"\"\"A simple function that does nothing.\"\"\"
    pass
""")
    yield repo_dir


def test_index_documents(test_db):
    """
    Unit test for the index_documents function.
    It checks if documents are correctly added to the database and if the call graph is built.
    """
    # The `test_db` fixture ensures a clean in-memory database.
    docs = load_and_split_repository("src") # Use our own src code for a quick test
    
    # Run the indexing process
    index_documents(docs)

    # 1. Check if CodeChunk objects were created
    assert CodeChunk.select().count() > 0, "No CodeChunk objects were created."

    # 2. Check if the CallGraph was built
    assert CallGraph.select().count() > 0, "CallGraph was not built."

    # 3. Verify a specific relationship
    # In src/db.py, build_call_graph calls get_all_code_chunks, json.loads, and create_call_graph_edge
    caller = select_code_chunk_by_name("build_call_graph", "src/db.py")
    dependencies = get_dependencies(caller)
    
    dep_names = {dep.name for dep in dependencies}
    assert "get_all_code_chunks" in dep_names
    assert "create_call_graph_edge" in dep_names


def test_load_and_split_repository(sample_repo):
    """
    Integration test for load_and_split_repository.
    """
    docs = load_and_split_repository(str(sample_repo))

    assert len(docs) == 2, "Expected to find two function documents."

    expected_names = {"func_a", "func_b"}
    found_names = {doc.metadata["name"] for doc in docs}

    assert expected_names == found_names, f"Missing or incorrect code structures found. Found: {found_names}"

    # Check that the call metadata is correct
    for doc in docs:
        if doc.metadata["name"] == "func_a":
            assert doc.metadata["calls"] == ["func_b"]
        if doc.metadata["name"] == "func_b":
            assert doc.metadata["calls"] == []
