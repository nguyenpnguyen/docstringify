import pytest
from peewee import SqliteDatabase
from langchain_core.documents import Document

from docstringify.db import (
    CodeChunk,
    CallGraph,
    get_or_create_code_chunk,
)

# A sample document representing a function that calls another function.
DOC_A = Document(
    page_content="def func_a(self):\n    \"\"\"Doc A\"\"\"\n    self.func_b()",
    metadata={
        "path": "test.py",
        "name": "func_a",
        "type": "method",
        "parent_class": "MyClass",
        "line_number": 1,
        "docstring": "Doc A",
        "calls": ["func_b"],
    },
)

# A sample document representing a function that is called.
DOC_B = Document(
    page_content="def func_b(self):\n    \"\"\"Doc B\"\"\"" ,
    metadata={
        "path": "test.py",
        "name": "func_b",
        "type": "method",
        "parent_class": "MyClass",
        "line_number": 5,
        "docstring": "Doc B",
        "calls": [],
    },
)

# A sample document representing a standalone function.
DOC_C = Document(
    page_content="def func_c():\n    \"\"\"Doc C\"\"\"" ,
    metadata={
        "path": "utils.py",
        "name": "func_c",
        "type": "function",
        "line_number": 1,
        "docstring": "Doc C",
        "calls": [],
    },
)


@pytest.fixture(autouse=True)
def test_db():
    """
    Pytest fixture to use an in-memory SQLite database for all tests.
    This automatically sets up and tears down the database for isolation.
    """
    test_db = SqliteDatabase(":memory:")
    models = [CodeChunk, CallGraph]

    # Temporarily bind the models to the in-memory database
    with test_db.bind_ctx(models):
        test_db.connect()
        test_db.create_tables(models)
        yield test_db
        test_db.drop_tables(models)
        test_db.close()


@pytest.fixture
def populated_db(test_db):
    """
    Fixture that populates the in-memory database with sample code chunks.
    Depends on the `test_db` fixture to ensure a clean database.
    """
    # Create the code chunks in the database
    get_or_create_code_chunk(DOC_A)
    get_or_create_code_chunk(DOC_B)
    get_or_create_code_chunk(DOC_C)

    return test_db
