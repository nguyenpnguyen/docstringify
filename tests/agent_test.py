import pytest
from langchain_core.documents import Document
from src.agent import retrieve_node, generate_node, agent, AgentState

# The `test_vector_store` fixture from conftest.py is used in these tests.

def test_retrieve_node(test_vector_store, monkeypatch):
    """
    Tests the retrieve_node to ensure it gets context from the vector store.
    It uses monkeypatch to replace the agent's global vector_store with the test fixture.
    """
    # This code snippet is intentionally similar to content in the test_vector_store
    code_snippet = "def add(x, y): return x + y"
    initial_state = AgentState(code_snippet=code_snippet)

    # Replace the vector_store used by the retrieve_node with our test fixture
    monkeypatch.setattr("src.agent.vector_store", test_vector_store)

    # The config argument is not used in the node, so we can pass a dummy
    result_state = retrieve_node(initial_state, config=None)

    # 1. Check that the 'context' field was populated
    assert "context" in result_state
    assert isinstance(result_state["context"], list)

    # 2. Check that we retrieved at least one document
    assert len(result_state["context"]) > 0, "Retrieve node failed to find any context."

    # 3. Check that the retrieved content is relevant
    assert "def add(a: int, b: int)" in result_state["context"][0].page_content


def test_generate_node():
    """
    Tests the generate_node to ensure it generates a docstring.
    This is an integration test as it calls the actual LLM.
    """
    code_snippet = "def my_func(a, b): return a * b"
    context_docs = [
        Document(page_content="def multiply(x, y): return x * y", metadata={"source": "math.py"})
    ]
    initial_state = AgentState(
        code_snippet=code_snippet,
        context=context_docs
    )

    # The config argument is not used in the node, so we can pass a dummy
    result_state = generate_node(initial_state, config=None)

    # Check that a docstring was generated
    assert "docstring" in result_state
    assert isinstance(result_state["docstring"], str)
    assert len(result_state["docstring"]) > 10, "Generated docstring is too short."
    assert "Args:" in result_state["docstring"], "Docstring should follow Google style with 'Args:'."


def test_agent_workflow(test_vector_store, monkeypatch):
    """
    Tests the full agent workflow from end to end.
    It uses monkeypatch to replace the agent's vector_store with our pre-populated test fixture.
    """
    # Replace the vector_store used by the agent with our test fixture
    monkeypatch.setattr("src.agent.vector_store", test_vector_store)

    code_snippet = "def subtract(x, y): return x - y"
    inputs = {"code_snippet": code_snippet}

    # Run the full agent workflow
    final_state = agent.invoke(inputs)

    # 1. Check for a generated docstring
    assert "docstring" in final_state
    assert final_state["docstring"] is not None
    assert len(final_state["docstring"]) > 10

    # 2. Check that context was retrieved
    assert "context" in final_state
    assert len(final_state["context"]) > 0
    assert "def subtract(a: int, b: int)" in final_state["context"][0].page_content
