import logging
from langgraph.graph import StateGraph, END
from docstringify.nodes import (
    ApplicationState,
    builder_node,
    dispatcher_node,
    retrieval_node,
    generation_node,
    patcher_node,
    final_writer_node,
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --- Graph Construction ---

def should_continue(state: ApplicationState):
    """
    Conditional edge logic that determines whether to continue processing
    or trigger the final write.
    """
    if state["current_job_id"] is None:
        return "final_write" # Trigger final write when queue is empty
    else:
        return "continue"

graph = StateGraph(ApplicationState)

# Add nodes
graph.add_node("builder", builder_node)
graph.add_node("dispatcher", dispatcher_node)
graph.add_node("retriever", retrieval_node)
graph.add_node("generator", generation_node)
graph.add_node("patcher", patcher_node)
graph.add_node("final_writer", final_writer_node)

# Define edges
graph.set_entry_point("builder")
graph.add_edge("builder", "dispatcher")
graph.add_conditional_edges(
    "dispatcher",
    should_continue,
    {
        "continue": "retriever",
        "final_write": "final_writer",
    },
)
graph.add_edge("retriever", "generator")
graph.add_edge("generator", "patcher")
graph.add_edge("patcher", "dispatcher")
graph.add_edge("final_writer", END)

# Compile the agent
workflow = graph.compile()
