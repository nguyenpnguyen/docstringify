import os
import logging

from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

# Use LangGraph for precise control over the "retrieve -> generate" flow
from langgraph.graph import StateGraph, END

# Import your custom modules (Assumed to exist based on your snippet)
from src.retrievers import retrieve_relevant_docs

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
# Load LLM and Embeddings
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

class ResponseFormat(BaseModel):
    """The structured response expected from the generation agent."""
    docstring: str


class AgentState(BaseModel):
    """The state of the agent execution pipeline."""
    code_snippet: str
    context: list[Document] = Field(default_factory=list)
    docstring: Optional[str] = None


# --- Prompts ---

# Updated prompt: removed tool instructions since context is now injected directly
SYSTEM_PROMPT = """
You are an expert Python documentation generator.
Your task is to generate a clear and concise Google-style docstring for the provided code snippet.
Use the context provided (retrieved from the codebase) to understand dependencies, types, and logic.

Output ONLY the docstring string.
"""


# --- Nodes ---

def retrieve_node(state: AgentState, config: RunnableConfig):
    """
    Node 1: Deterministic Retrieval
    Calls the python function directly to get context.
    """
    code = state.code_snippet

    docs = retrieve_relevant_docs(code)

    return {"context": docs}


def generate_node(state: AgentState, config: RunnableConfig):
    """
    Node 2: Generation
    Uses LLM with Structured Output to generate the docstring.
    """
    code = state.code_snippet
    context_docs = state.context

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
    structured_llm = llm.with_structured_output(ResponseFormat, method="json_schema")
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
