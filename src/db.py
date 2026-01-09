from peewee import *
from langchain_core.documents import Document

db = SqliteDatabase(':memory:', pragmas={'journal_mode': 'wal'})

class BaseModel(Model):
    class Meta:
        database = db

class CodeChunk(BaseModel):
    path = TextField()
    name = TextField()
    type = TextField()
    content = TextField()
    parent_class = TextField(null=True)
    line_number = IntegerField()
    docstring = TextField(null=True)

class CallGraphEdge(BaseModel):
    caller = ForeignKeyField(CodeChunk, backref='outgoing_calls')
    callee = TextField()


def create_code_chunk(code_chunk: Document) -> CodeChunk:
    return CodeChunk.create(path=code_chunk.metadata['path'],
                     name=code_chunk.metadata['name'],
                     type=code_chunk.metadata['type'],
                     content=code_chunk.page_content,
                     parent_class=code_chunk.metadata.get('parent_class'),
                     line_number=code_chunk.metadata['line_number'],
                     docstring=code_chunk.metadata.get('docstring'))

def select_code_chunk_by_name(name: str) -> CodeChunk:
    return CodeChunk.get(CodeChunk.name == name)

def update_code_chunk(code_chunk: CodeChunk, **kwargs):
    if 'name' in kwargs:
        code_chunk.name = kwargs['name']
    if 'path' in kwargs:
        code_chunk.path = kwargs['path']
    if 'type' in kwargs:
        code_chunk.type = kwargs['type']
    if 'content' in kwargs:
        code_chunk.content = kwargs['content']
    if 'parent_class' in kwargs:
        code_chunk.parent_class = kwargs['parent_class']
    if 'line_number' in kwargs:
        code_chunk.line_number = kwargs['line_number']
    if 'docstring' in kwargs:
        code_chunk.docstring = kwargs['docstring']

    code_chunk.save()

def create_call_graph_edge(caller: CodeChunk, callee: str) -> CallGraphEdge:
    return CallGraphEdge.create(caller=caller, callee=callee)
