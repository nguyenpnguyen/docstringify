import pytest
from unittest.mock import patch
from langchain_core.documents import Document

from src.agent import retrieve_node, generate_node, agent, AgentState
from src.db import build_call_graph, select_code_chunk_by_name


def test_retrieve_node(populated_db):
    """
    Tests the retrieve_node to ensure it gets context by traversing the call graph.
    """
    # First, build the graph from the data in the populated_db
    build_call_graph()

    # The initial state contains the name of the chunk we want to process
    initial_state = AgentState(code_chunk_name="func_a")

    # Run the retrieve node
    result_state = retrieve_node(initial_state)

    # 1. Check that the 'context' field was populated
    assert "context" in result_state
    assert isinstance(result_state["context"], list)
    assert len(result_state["context"]) > 0, "Retrieve node failed to find any context."

    # 2. Check that the retrieved content is correct
    # The context for 'func_a' should be 'func_b' (its dependency).
    context_names = {doc.metadata["name"] for doc in result_state["context"]}
    assert "func_b" in context_names


def test_generate_node():
    """
    Tests the generate_node to ensure it generates a docstring.
    This is an integration test as it calls the actual LLM.
    """
    code_chunk = select_code_chunk_by_name("func_a") # Assume populated_db is run
    context_docs = [
        Document(page_content="def func_b(self): pass", metadata={"name": "func_b"})
    ]

    initial_state = AgentState(
        code_chunk_name="func_a",
        code_chunk=code_chunk,
        context=context_docs
    )

    # The config argument is not used in the node, so we can pass a dummy
    result_state = generate_node(initial_state)

    # Check that a docstring was generated
    assert "docstring" in result_state
    assert isinstance(result_state["docstring"], str)
    assert len(result_state["docstring"]) > 10, "Generated docstring is too short."
    assert "Args:" in result_state["docstring"], "Docstring should follow Google style with 'Args:'."


def test_agent_workflow(populated_db):
    """
    Tests the full agent workflow with the new database-backed retrieval.
    """
    # Build the call graph from the test data
    build_call_graph()

    # The input to the agent is the name of the function to document
    inputs = {"code_chunk_name": "func_a"}

    # Run the full agent workflow
    final_state = agent.invoke(inputs)

    # 1. Check for a generated docstring
    assert "docstring" in final_state
    assert final_state["docstring"] is not None
    assert len(final_state["docstring"]) > 10

    # 2. Check that the correct code chunk was processed
    assert final_state["code_chunk"].name == "func_a"

    # 3. Check that context was retrieved from the database
    assert "context" in final_state
    assert len(final_state["context"]) > 0
    assert final_state["context"][0].metadata["name"] == "func_b"
