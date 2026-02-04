import os
import logging
from dataclasses import dataclass
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
import re
from langchain_ollama import ChatOllama
from src.db import init_db, get_undocumented_chunks, bulk_insert_chunks, build_call_graph, select_code_chunk_by_id, update_code_chunk_docstring
from src.retrievers import retrieve_relevant_docs
from src.injector import load_and_split_repository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_name = os.path.basename(current_dir)

@dataclass
class LLMConfig:
    llm_id: str = "qwen3:4b-instruct"
    temperature: float = 0.2
    num_ctx: int = 8192

llm_cfg = LLMConfig()

# --- Initialization ---
try:
    llm = ChatOllama(
        model=llm_cfg.llm_id,
        temperature=llm_cfg.temperature,
        validate_model_on_init=True,
        num_ctx=llm_cfg.num_ctx,
    )
except Exception as e:
    logger.error(f"Error loading LLM '{llm_cfg.llm_id}': {e}")

# --- Schemas ---
class AgentState(TypedDict):
    """The state of the agent execution pipeline."""
    db_path: str
    queue: List[int]
    current_job_id: Optional[int]
    current_code: Optional[str]
    retrieved_context: str
    generated_docstring: Optional[str]


# --- Prompts ---

SYSTEM_PROMPT = """
You are an expert Python documentation generator.
Your task is to generate a clear and concise Google-style docstring for the provided code snippet.
Use the context provided (retrieved from the codebase) to understand dependencies, types, and logic.

Output ONLY the docstring string.
"""

# --- Nodes ---

def builder_node(state: AgentState):
    """
    Node A: Builder
    Checks if the database exists. If not, it parses the repository,
    populates the database, and builds the call graph. Then, it queries
    the database for undocumented functions and populates the agent's queue.
    """
    db_path = state["db_path"]
    if not os.path.exists(db_path):
        # Database does not exist, so we need to build it.
        init_db()
        
        # Infer the repository path from the db_path location, assuming it's in the repo root
        repo_path = os.path.dirname(os.path.abspath(db_path))
        
        # Load and parse the repository content
        docs = load_and_split_repository(repo_path)
        
        # Insert the parsed documents into the database
        bulk_insert_chunks(docs)
        
        # Build the call graph from the inserted data
        build_call_graph()

    # Query for undocumented functions to populate the queue
    undocumented_chunks = get_undocumented_chunks()
    queue = [chunk.id for chunk in undocumented_chunks]
    
    return {"queue": queue}


def dispatcher_node(state: AgentState):
    """
    Node B: Dispatcher
    Checks the queue of undocumented functions. If the queue is not empty,
    it pops the next function ID, retrieves its data from the database,
    and updates the agent's state. If the queue is empty, it signals the end.
    """
    queue = state["queue"]
    if not queue:
        return {"current_job_id": None, "current_code": None}
    
    # Pop the first ID from the queue
    job_id = queue.pop(0)
    
    # Fetch the code chunk from the database
    code_chunk = select_code_chunk_by_id(job_id)
    
    # Update the state
    return {
        "queue": queue,
        "current_job_id": job_id,
        "current_code": code_chunk.content,
    }


def retrieval_node(state: AgentState):
    """
    Node C: Retrieval (SQL-RAG)
    Retrieves context for the current job by querying the database for
    dependencies and usages, then formats them into a single string.
    """
    job_id = state["current_job_id"]
    code_chunk = select_code_chunk_by_id(job_id)
    
    # Retrieve relevant documents (dependencies and usages)
    docs = retrieve_relevant_docs(code_chunk.name, code_chunk.path)
    
    # Format the retrieved context into a string
    context_str = "\n\n".join(
        [f"--- Source: {d.metadata.get('path', 'unknown')} ---\n{d.page_content}"
         for d in docs]
    )
    
    return {"retrieved_context": context_str}


def generation_node(state: AgentState):
    """
    Node D: Generation
    Uses the LLM to generate a docstring for the current code snippet,
    based on the retrieved context.
    """
    code = state["current_code"]
    context_str = state["retrieved_context"]
    
    # Construct messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"### Context from Codebase:\n{context_str}\n\n### Target Code:\n{code}")
    ]
    
    # Invoke the LLM
    response = llm.invoke(messages)
    
    # Extract the docstring from the response
    docstring = response.content.strip().strip('"""').strip("'''")
    
    return {"generated_docstring": docstring}


def get_indentation(line: str) -> str:
    """Extracts the leading whitespace from a line."""
    match = re.match(r"^(\s*)", line)
    return match.group(1) if match else ""

def patcher_node(state: AgentState):
    """
    Node E: Patcher
    Inserts the generated docstring into the source code file at the
    correct line number and with the correct indentation. It also updates
    the database to mark the function as documented.
    """
    job_id = state["current_job_id"]
    docstring = state["generated_docstring"]
    
    # Get the code chunk details from the database
    code_chunk = select_code_chunk_by_id(job_id)
    
    # Read the source file
    with open(code_chunk.path, "r") as f:
        lines = f.readlines()
        
    # Get the indentation of the function definition
    func_line = lines[code_chunk.line_number - 1]
    indentation = get_indentation(func_line) + "    " # Add 4 spaces for the docstring
    
    # Format the docstring with the correct indentation
    formatted_docstring = f'{indentation}"""{docstring}"""\n'
    
    # Insert the docstring into the file content
    lines.insert(code_chunk.line_number, formatted_docstring)
    
    # Write the modified content back to the file
    with open(code_chunk.path, "w") as f:
        f.writelines(lines)
        
    # Update the database
    update_code_chunk_docstring(code_chunk, "DONE")
    
    # Clear the current job ID
    return {"current_job_id": None}


# --- Graph Construction ---

def should_continue(state: AgentState):
    """
    Conditional edge logic that determines whether to continue processing
    or end the workflow.
    """
    if state["current_job_id"] is None:
        return "end"
    else:
        return "continue"

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("builder", builder_node)
workflow.add_node("dispatcher", dispatcher_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("generator", generation_node)
workflow.add_node("patcher", patcher_node)

# Define edges
workflow.set_entry_point("builder")
workflow.add_edge("builder", "dispatcher")
workflow.add_conditional_edges(
    "dispatcher",
    should_continue,
    {
        "continue": "retriever",
        "end": END,
    },
)
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "patcher")
workflow.add_edge("patcher", "dispatcher")

# Compile the agent
agent = workflow.compile()
