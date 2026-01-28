from src.db import (
    build_call_graph,
    get_dependencies,
    get_dependents,
    select_code_chunk_by_name,
    CodeChunk,
)


def test_build_call_graph(populated_db):
    """
    Tests that the build_call_graph function correctly populates the CallGraph table.
    """
    # The populated_db fixture creates chunks, but the graph is not built yet.
    # Now, build the call graph.
    build_call_graph()

    # Retrieve the caller chunk from the DB
    caller_chunk = select_code_chunk_by_name("func_a")

    # Check its outgoing calls (dependencies)
    dependencies = get_dependencies(caller_chunk)
    assert len(dependencies) == 1
    assert dependencies[0].name == "func_b"

    # Now check the other direction for the callee
    callee_chunk = select_code_chunk_by_name("func_b")
    dependents = get_dependents(callee_chunk)
    assert len(dependents) == 1
    assert dependents[0].name == "func_a"


def test_get_dependencies(populated_db):
    """
    Tests the get_dependencies function.
    """
    # First, we need to manually build the graph for this test
    build_call_graph()

    func_a = select_code_chunk_by_name("func_a")
    func_b = select_code_chunk_by_name("func_b")
    func_c = select_code_chunk_by_name("func_c")

    # func_a calls func_b
    deps_a = get_dependencies(func_a)
    assert len(deps_a) == 1
    assert deps_a[0].name == "func_b"

    # func_b and func_c don't call anything
    deps_b = get_dependencies(func_b)
    assert len(deps_b) == 0

    deps_c = get_dependencies(func_c)
    assert len(deps_c) == 0


def test_get_dependents(populated_db):
    """
    Tests the get_dependents function.
    """
    # First, we need to manually build the graph for this test
    build_call_graph()

    func_a = select_code_chunk_by_name("func_a")
    func_b = select_code_chunk_by_name("func_b")
    func_c = select_code_chunk_by_name("func_c")

    # func_b is called by func_a
    dependents_b = get_dependents(func_b)
    assert len(dependents_b) == 1
    assert dependents_b[0].name == "func_a"

    # func_a and func_c are not called by anything in our test set
    dependents_a = get_dependents(func_a)
    assert len(dependents_a) == 0

    dependents_c = get_dependents(func_c)
    assert len(dependents_c) == 0


def test_code_chunk_creation(populated_db):
    """
    Tests that code chunks are created correctly by the populated_db fixture.
    """
    chunks = CodeChunk.select()
    assert chunks.count() == 3

    # Verify one of the chunks
    func_a = select_code_chunk_by_name("func_a")
    assert func_a.path == "test.py"
    assert func_a.parent_class == "MyClass"
    assert '["func_b"]' in func_a.calls
