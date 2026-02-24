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
    level=logging.INFO,
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

workflow = StateGraph(ApplicationState)

# Add nodes
workflow.add_node("builder", builder_node)
workflow.add_node("dispatcher", dispatcher_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("generator", generation_node)
workflow.add_node("patcher", patcher_node)
workflow.add_node("final_writer", final_writer_node)

# Define edges
workflow.set_entry_point("builder")
workflow.add_edge("builder", "dispatcher")
workflow.add_conditional_edges(
    "dispatcher",
    should_continue,
    {
        "continue": "retriever",
        "final_write": "final_writer",
    },
)
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "patcher")
workflow.add_edge("patcher", "dispatcher")
workflow.add_edge("final_writer", END)

# Compile the agent
app = workflow.compile()
