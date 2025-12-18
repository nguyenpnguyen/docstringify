import sqlite3, sqlite_vec
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from src.loader import load_llm, load_embeddings, load_vector_store

from dataclasses import dataclass
from typing import List
from langchain.agents import create_agent
from src.retrievers import retrieve_context
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_core.documents import Document
from langchain.agents.structured_output import ToolStrategy

LLM_ID = "qwen3:4b-instruct"
EMBED_ID = "qwen3-embedding:0.6b"
TABLE = "eval_table"
DB_FILE = "eval_vec.db"

llm: BaseChatModel = load_llm(model=LLM_ID, temperature=0.2, num_ctx=8192)

embeddings: Embeddings = load_embeddings(model=EMBED_ID)

db = sqlite3.connect(database=DB_FILE)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

vector_store: VectorStore = load_vector_store(embedding=embeddings, connection=db, table=TABLE, db_file=DB_FILE)

SYSTEM_PROMPT = """
You are an expert Python documentation generator.
Below is a python function or class that needs a docstring.
You have access to a tool:
 
- retrieve_context: use this to retrieve relevant code snippets from a large codebase to help you understand the context of the target code.

Use the tool as needed to gather context, then generate a Google-style docstring for the target code.
"""


@dataclass
class Context:
    code_snippet: str
    context: List[Document]


@dataclass
class ResponseFormat:
    docstring: str


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
)
