import os
import logging
import re
from typing import TypedDict, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from docstringify.db import (
    init_db,
    get_undocumented_chunks,
    bulk_insert_chunks,
    build_call_graph,
    select_code_chunk_by_id,
    update_code_chunk_docstring,
    get_all_code_chunks,
)
from docstringify.retrievers import retrieve_relevant_docs
from docstringify.injector import load_and_split_repository
from docstringify.config import settings
from docstringify.utils import get_indentation, find_docstring_boundaries

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --- Schemas ---
class ApplicationState(TypedDict):
    """The state of the agent execution pipeline."""
    db_path: str
    queue: list[int]
    current_job_id: Optional[int]
    current_code: Optional[str]
    retrieved_context: str
    generated_docstring: Optional[str]
    file_docstring_changes: dict[str, list[tuple[int, str, int, str]]] # Updated to store (insert_line, docstring, body_start_line, indentation)


# --- Prompts ---

SYSTEM_PROMPT = """
You are an expert Python documentation generator.
Your task is to generate a clear and concise Google-style docstring for the provided Python code snippet.
Use the context provided (retrieved from the codebase) to understand dependencies, types, and logic.

Crucially, you MUST ONLY output the docstring content itself.
DO NOT include the triple quotes in your response.
DO NOT include the function/class definition or any other code.
DO NOT wrap your response in markdown code blocks (e.g., '```python').
Your output should be solely the text that would appear *between* the opening and closing triple quotes of a docstring.
"""

# --- Nodes ---

def builder_node(state: ApplicationState):
    logger.info(">>> Executing builder_node...")
    """
    Node A: Builder
    Checks if the database is initialized and has entries. If not, it parses the
    repository, populates the database, and builds the call graph. Then, it
    queries the database for undocumented functions and populates the agent's queue.
    """
    db_path = state["db_path"]
    logger.info(f"db_path: {db_path}")
    
    # Ensure database is initialized
    init_db(db_path)
    
    # Check if the database has any chunks; if not, populate it
    all_chunks = get_all_code_chunks()
    logger.debug(f"Initial chunks count: {len(all_chunks)}")
    
    if not all_chunks:
        # Infer the repository path from the db_path location, assuming it's in the repo root
        repo_path = os.path.dirname(os.path.abspath(db_path))
        logger.debug(f"repo_path: {repo_path}")
        
        logger.debug(f"Indexing repository at {repo_path}...")
        # Load and parse the repository content
        docs = load_and_split_repository(repo_path)
        logger.debug(f"Docs found during indexing: {len(docs)}")
        
        # Insert the parsed documents into the database
        bulk_insert_chunks(docs)
        
        # Build the call graph from the inserted data
        build_call_graph()

    # Query for undocumented functions to populate the queue
    undocumented_chunks = get_undocumented_chunks()
    logger.info(f"Number of undocumented chunks found: {len(undocumented_chunks)}")
    queue = [chunk.id for chunk in undocumented_chunks]
    
    return {"queue": queue}


def dispatcher_node(state: ApplicationState):
    logger.info(">>> Executing dispatcher_node...")
    """
    Node B: Dispatcher
    Checks the queue of undocumented functions. If the queue is not empty,
    it pops the next function ID, retrieves its data from the database,
    and updates the agent's state. If the queue is empty, it signals the end.
    """
    queue = state["queue"]
    logger.info(f"Queue size: {len(queue)}")
    if not queue:
        logger.info("Queue is empty, finishing.")
        return {"current_job_id": None, "current_code": None}
    
    # Pop the first ID from the queue
    job_id = queue.pop(0)
    logger.info(f"Dispatching job_id: {job_id}")
    
    # Fetch the code chunk from the database
    code_chunk = select_code_chunk_by_id(job_id)
    logger.debug(f"Working on chunk: {code_chunk.name} in {code_chunk.path}")
    
    # Update the state
    return {
        "queue": queue,
        "current_job_id": job_id,
        "current_code": code_chunk.content,
    }


def retrieval_node(state: ApplicationState):
    logger.info("Executing retrieval_node...")
    """
    Node C: Retrieval (SQL-RAG)
    Retrieves context for the current job by querying the database for
    dependencies and usages, then formats them into a single string.
    """
    job_id = state["current_job_id"]
    code_chunk = select_code_chunk_by_id(job_id)
    logger.debug(f">>> Retrieving context for: {code_chunk.name}")
    
    # Retrieve relevant documents (dependencies and usages)
    docs = retrieve_relevant_docs(code_chunk.name, code_chunk.path)
    logger.debug(f">>> Retrieved {len(docs)} context documents.")
    
    # Format the retrieved context into a string
    context_str = "\n\n".join(
        [f"--- Source: {d.metadata.get('path', 'unknown')} ---\n{d.page_content}"
         for d in docs]
    )
    
    return {"retrieved_context": context_str}


def generation_node(state: ApplicationState):
    logger.info("Executing generation_node...")
    """
    Node D: Generation
    Uses the LLM to generate a docstring for the current code snippet,
    based on the retrieved context.
    """
    code = state["current_code"]
    context_str = state["retrieved_context"]
    job_id = state["current_job_id"]
    code_chunk = select_code_chunk_by_id(job_id)
    
    logger.debug(f"Generating docstring for: {code_chunk.name}")
    logger.debug(f"Context passed to LLM:\n{context_str}")
    
    # Initialize the LLM with the current settings
    llm = ChatOllama(
        model=settings.llm_id,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
    )
    
    # Construct messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"### Context from Codebase:\n{context_str}\n\n### Target Code:\n{code}")
    ]
    
    # Invoke the LLM
    response = llm.invoke(messages)
    
    # Extract the docstring from the response using robust post-processing
    raw_response = response.content
    
    # Attempt to extract content between triple quotes, ignoring code blocks
    # This regex looks for """ (or ''') followed by content, and then the closing """ (or ''')
    # It tries to avoid matching ```python...``` blocks
    match = re.search(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', raw_response, re.DOTALL)
    if match:
        docstring = match.group(1).strip()
    else:
        # Fallback if regex doesn't find it, perhaps it's just raw content without quotes
        docstring = raw_response.strip()

    # Further clean up any leftover markdown code block markers
    docstring = re.sub(r'```python\n?', '', docstring, flags=re.IGNORECASE)
    docstring = re.sub(r'\n?```', '', docstring)

    # Remove any leading/trailing triple quotes that might have survived
    docstring = docstring.strip().strip('"""').strip("'''")
    
    return {"generated_docstring": docstring}


def patcher_node(state: ApplicationState):
    logger.info("Executing patcher_node...")
    """
    Node E: Patcher
    Collects the generated docstring and its insertion point, storing them
    in the agent's state for later writing to the file. It also updates
    the database to mark the function as documented.
    """
    job_id = state["current_job_id"]
    docstring = state["generated_docstring"]
    
    # Get the code chunk details from the database
    code_chunk = select_code_chunk_by_id(job_id)
    logger.debug(f"Patching docstring for: {code_chunk.name}")
    
    # Read the source file (locally for calculation, not writing yet)
    with open(code_chunk.path, "r") as f:
        lines = f.readlines()
        
    # Get the indentation of the function definition
    func_indentation = get_indentation(lines[code_chunk.line_number - 1])

    # Find existing docstring and calculate new insert_line
    docstring_start, docstring_end = find_docstring_boundaries(lines, code_chunk.body_start_line, func_indentation)

    insert_line = code_chunk.body_start_line
    # If an existing docstring is found, the new docstring will overwrite it.
    # The insert_line will be the start of the existing docstring.
    if docstring_start is not None and docstring_end is not None:
        insert_line = docstring_start + 1 # Convert back to 1-based for consistent storage
        
    docstring_block_indentation = func_indentation + "    " # The indentation for the triple quotes and the start of the docstring content
    
    # Split the docstring into lines and format each line with the correct indentation
    formatted_docstring_lines = []
    formatted_docstring_lines.append(f'{docstring_block_indentation}"""')
    
    for line in docstring.split('\n'):
        if line.strip(): # If the line has content
            formatted_docstring_lines.append(f'{docstring_block_indentation}{line}')
        else: # Keep empty lines as is, but still ensure they have the block indentation
            formatted_docstring_lines.append(docstring_block_indentation)
            
    formatted_docstring_lines.append(f'{docstring_block_indentation}"""')
    formatted_docstring_lines.append('\n')

    formatted_docstring = "\n".join(formatted_docstring_lines)

    # Store the modification in the state, grouped by file path
    if code_chunk.path not in state["file_docstring_changes"]:
        state["file_docstring_changes"][code_chunk.path] = []
    # Store body_start_line and func_indentation so final_writer can re-find boundaries correctly
    state["file_docstring_changes"][code_chunk.path].append((insert_line, formatted_docstring, code_chunk.body_start_line, func_indentation))
        
    # Update the database
    update_code_chunk_docstring(code_chunk, "DONE")
    
    # Clear the current job ID
    return {"current_job_id": None}


def final_writer_node(state: ApplicationState):
    logger.info("Executing final_writer_node...")
    """
    Node F: Final Writer
    Writes all accumulated docstring changes to their respective files.
    Changes are applied in reverse order of line number to prevent
    issues with line shifts during insertion.
    """
    for file_path, changes in state["file_docstring_changes"].items():
        if not changes:
            continue
        
        logger.debug(f"Writing docstring changes to: {file_path}")

        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Sort changes by line number in descending order
        changes.sort(key=lambda x: x[0], reverse=True)

        for insert_line, formatted_docstring, body_start_line, func_indentation in changes:
            # We need to re-find the existing docstring boundaries just before writing,
            # in case multiple docstrings in the same file are being processed
            # and previous insertions/deletions have altered line numbers.
            
            # Find existing docstring and remove it
            docstring_start, docstring_end = find_docstring_boundaries(lines, insert_line, func_indentation)

            if docstring_start is not None and docstring_end is not None:
                del lines[docstring_start : docstring_end + 1]
                actual_insert_line = docstring_start
            else:
                actual_insert_line = insert_line - 1
            
            lines.insert(actual_insert_line, formatted_docstring)
            
        with open(file_path, "w") as f:
            f.writelines(lines)
            
    # Clear changes from state after writing
    return {"file_docstring_changes": {}}
