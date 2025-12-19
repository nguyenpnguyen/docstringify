import sqlite3
import sqlite_vec
from typing import List
from dataclasses import dataclass

from langchain_community.vectorstores.sqlitevec import SQLiteVec
from pydantic import BaseModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Use LangGraph for precise control over the "retrieve -> generate" flow
from langgraph.graph import StateGraph, END

# Import your custom modules (Assumed to exist based on your snippet)
from src.loader import load_llm, load_embeddings, load_vector_store
from src.retrievers import retrieve_relevant_docs

# --- Configuration ---
LLM_ID = "qwen3:4b-instruct"
EMBED_ID = "qwen3-embedding:0.6b"
TABLE = "eval_table"
DB_FILE = "eval_vec.db"

# --- Initialization ---
# Load LLM and Embeddings
llm: BaseChatModel = load_llm(model=LLM_ID, temperature=0.2, num_ctx=8192)
embeddings: Embeddings = load_embeddings(model=EMBED_ID)

db = SQLiteVec.create_connection(db_file=DB_FILE)

vector_store: VectorStore = load_vector_store(
    embedding=embeddings,
    connection=db,
    table=TABLE,
    db_file=DB_FILE
)


# --- Schemas ---

@dataclass
class ResponseFormat:
    """The structured response expected from the generation agent."""
    docstring: str


class AgentState(BaseModel):
    """The state of the agent execution pipeline."""
    code_snippet: str
    context: List[Document]
    docstring: str


# --- Prompts ---

# Updated prompt: removed tool instructions since context is now injected directly
SYSTEM_PROMPT = """
You are an expert Python documentation generator.
Your task is to generate a Google-style docstring for the provided code snippet.
Use the context provided (retrieved from the codebase) to understand dependencies, types, and logic.

Output ONLY the docstring string.
"""


# --- Nodes ---

def retrieve_node(state: AgentState, config: RunnableConfig):
    """
    Node 1: Deterministic Retrieval
    Calls the python function directly to get context.
    """
    code = state["code_snippet"]

    # We call the python function directly as requested.
    # Assuming retrieve_relevant_docs takes (query, vector_store) or similar.
    # Adjust arguments based on your actual src.tools implementation.
    docs = retrieve_relevant_docs(code, vector_store)

    return {"context": docs}


def generate_node(state: AgentState, config: RunnableConfig):
    """
    Node 2: Generation
    Uses LLM with Structured Output to generate the docstring.
    """
    code = state["code_snippet"]
    context_docs = state["context"]

    # Format context for the prompt
    context_str = "\n\n".join(
        [f"--- Source: {d.metadata.get('source', 'unknown')} ---\n{d.page_content}"
         for d in context_docs]
    )

    # Construct messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"### Context from Codebase:\n{context_str}\n\n### Target Code:\n{code}")
    ]

    # Bind structured output to force the ResponseFormat
    structured_llm = llm.with_structured_output(ResponseFormat)
    response = structured_llm.invoke(messages)

    return {"docstring": response.docstring}


# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Define edges: Start -> Retrieve -> Generate -> End
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the agent
agent = workflow.compile()
