import json
from peewee import *
from langchain_core.documents import Document

# Use a Proxy for dynamic database initialization
db = Proxy()


class BaseModel(Model):
    class Meta:
        database = db


class CodeChunk(BaseModel):
    path = TextField()
    name = TextField(unique=True)  # Assuming name is a unique identifier
    type = TextField()
    content = TextField()
    parent_class = TextField(null=True)
    line_number = IntegerField()
    body_start_line = IntegerField(null=True)
    docstring = TextField(null=True)
    calls = TextField(default="[]")  # Store as JSON string


class CallGraph(BaseModel):
    caller = ForeignKeyField(CodeChunk, backref="outgoing_calls")
    # Using TextField for callee allows for flexibility, e.g., calls to
    # external libraries or functions not yet parsed.
    # The alternative is a ForeignKeyField to CodeChunk, which would enforce
    # relational integrity but require callee chunks to exist first.
    callee = TextField()

    class Meta:
        # Ensure a call edge is unique
        primary_key = CompositeKey("caller", "callee")


def init_db(db_path: str):
    """Initializes the database and creates tables."""
    real_db = SqliteDatabase(db_path, pragmas={"journal_mode": "wal"})
    db.initialize(real_db)
    with db:
        db.create_tables([CodeChunk, CallGraph])


def get_or_create_code_chunk(code_chunk_doc: Document) -> tuple[CodeChunk, bool]:
    """Gets or creates a code chunk, storing its call data."""
    return CodeChunk.get_or_create(
        name=code_chunk_doc.metadata["name"],
        defaults={
            "path": code_chunk_doc.metadata["path"],
            "type": code_chunk_doc.metadata["type"],
            "content": code_chunk_doc.page_content,
            "parent_class": code_chunk_doc.metadata.get("parent_class"),
            "line_number": code_chunk_doc.metadata["line_number"],
            "body_start_line": code_chunk_doc.metadata.get("body_start_line"),
            "docstring": code_chunk_doc.metadata.get("docstring"),
            "calls": json.dumps(code_chunk_doc.metadata.get("calls", [])),
        },
    )


def bulk_insert_chunks(chunks: list[Document]):
    """
    Efficiently inserts or updates multiple code chunks.
    """
    with db.atomic():
        for chunk_doc in chunks:
            get_or_create_code_chunk(chunk_doc)


def get_undocumented_chunks() -> list[CodeChunk]:
    """
    Retrieves all code chunks that do not have a docstring.
    """
    return list(CodeChunk.select().where(CodeChunk.docstring.is_null()))

def select_code_chunk_by_name(name: str, path) -> CodeChunk:
    """Selects a code chunk by its unique name."""
    return CodeChunk.get((CodeChunk.name == name) & (CodeChunk.path == path))


def select_code_chunk_by_id(chunk_id: int) -> CodeChunk:
    """Selects a code chunk by its unique id."""
    return CodeChunk.get(CodeChunk.id == chunk_id)

def update_code_chunk_docstring(code_chunk: CodeChunk, docstring: str):
    """Updates the docstring of a specific code chunk."""
    query = CodeChunk.update(docstring=docstring).where(CodeChunk.id == code_chunk.id)
    query.execute()


def create_call_graph_edge(caller: CodeChunk, callee_name: str) -> CallGraph:
    """Creates a directed edge in the call graph."""
    return CallGraph.get_or_create(caller=caller, callee=callee_name)[0]


def get_dependencies(code_chunk: CodeChunk) -> list[CodeChunk]:
    """
    Retrieves all code chunks that the given code_chunk calls (its callees).
    """
    callee_names_query = CallGraph.select(CallGraph.callee).where(
        CallGraph.caller == code_chunk
    )
    callee_names = [edge.callee for edge in callee_names_query]

    if not callee_names:
        return []

    return list(CodeChunk.select().where(CodeChunk.name.in_(callee_names)))


def get_dependents(code_chunk: CodeChunk) -> list[CodeChunk]:
    """
    Retrieves all code chunks that call the given code_chunk (its callers).
    """
    return list(
        CodeChunk.select()
        .join(CallGraph, on=(CallGraph.caller == CodeChunk.id))
        .where(CallGraph.callee == code_chunk.name)
    )


def get_all_code_chunks() -> list[CodeChunk]:
    """Retrieves all code chunks from the database."""
    return list(CodeChunk.select())


def build_call_graph():
    """
    Builds the call graph from the 'calls' data stored in each code chunk.
    """
    all_chunks = get_all_code_chunks()

    with db.atomic():
        for chunk in all_chunks:
            callee_names = json.loads(chunk.calls)
            for callee_name in callee_names:
                create_call_graph_edge(chunk, callee_name)
