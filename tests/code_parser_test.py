import pytest
from src.code_parser import CodeStructureVisitor, parse_code_structure, split_code_by_length, get_splitter
from langchain_core.documents import Document
from unittest.mock import MagicMock
from langchain_text_splitters import PythonCodeTextSplitter

# Sample code for testing
CODE_SAMPLE_FUNCTIONS = """
def func_a():
    \"\"\"Doc A.\"\"\"
    pass

def func_b(arg1, arg2):
    \"\"\"Doc B.\"\"\"
    return arg1 + arg2
"""

CODE_SAMPLE_CLASS = """
class MyClass:
    \"\"\"Class doc.\"\"\"
    def __init__(self, val):
        self.val = val

    def method_a(self):
        \"\"\"Method A.\"\"\"
        return self.val * 2

    @property
    def prop(self):
        \"\"\"Property doc.\"\"\"
        return self.val
"""

CODE_SAMPLE_LONG_FUNCTION = """
def long_function():
    \"\"\"This is a very long docstring for a very long function.
    It has many lines of code to test the splitter functionality.
    \"\"\"
    # Line 1
    # Line 2
    # ... many lines
    # Line 100
    pass
""" * 10 # Make it artificially long

FILE_PATH = "test_file.py"

def test_code_structure_visitor_functions():
    visitor = CodeStructureVisitor(CODE_SAMPLE_FUNCTIONS, FILE_PATH)
    parse_code_structure(visitor, CODE_SAMPLE_FUNCTIONS, FILE_PATH)

    assert len(visitor.raw_documents) == 2
    doc_a = visitor.raw_documents[0]
    doc_b = visitor.raw_documents[1]

    assert "func_a" in doc_a.page_content
    assert doc_a.metadata["name"] == "func_a"
    assert doc_a.metadata["type"] == "function"
    assert doc_a.metadata["docstring"] == "Doc A."

    assert "func_b" in doc_b.page_content
    assert doc_b.metadata["name"] == "func_b"
    assert doc_b.metadata["type"] == "function"
    assert doc_b.metadata["docstring"] == "Doc B."

def test_code_structure_visitor_class():
    visitor = CodeStructureVisitor(CODE_SAMPLE_CLASS, FILE_PATH)
    parse_code_structure(visitor, CODE_SAMPLE_CLASS, FILE_PATH)

    # Expecting: Class definition (including __init__), method_a, prop
    assert len(visitor.raw_documents) == 3

    class_doc = visitor.raw_documents[0]
    method_a_doc = visitor.raw_documents[1]
    prop_doc = visitor.raw_documents[2]

    assert "class MyClass" in class_doc.page_content
    assert "__init__" in class_doc.page_content # __init__ is part of class chunk
    assert "method_a" not in class_doc.page_content # method_a is separate
    assert class_doc.metadata["name"] == "MyClass"
    assert class_doc.metadata["type"] == "class_definition"
    assert class_doc.metadata["docstring"] == "Class doc."
    assert class_doc.metadata["parent_class"] is None

    assert "def method_a" in method_a_doc.page_content
    assert method_a_doc.metadata["name"] == "method_a"
    assert method_a_doc.metadata["type"] == "method"
    assert method_a_doc.metadata["docstring"] == "Method A."
    assert method_a_doc.metadata["parent_class"] == "MyClass"

    assert "def prop" in prop_doc.page_content
    assert prop_doc.metadata["name"] == "prop"
    assert prop_doc.metadata["type"] == "method"
    assert prop_doc.metadata["docstring"] == "Property doc."
    assert prop_doc.metadata["parent_class"] == "MyClass"


def test_parse_code_structure_syntax_error():
    malformed_code = "def bad_func(:"
    visitor = CodeStructureVisitor(malformed_code, FILE_PATH)
    with pytest.raises(SyntaxError):
        parse_code_structure(visitor, malformed_code, FILE_PATH)

def test_get_splitter():
    splitter = get_splitter(chunk_size=100, chunk_overlap=10)
    assert isinstance(splitter, PythonCodeTextSplitter)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 10

def test_split_code_by_length():
    # Mock a visitor with a raw document that's too long
    visitor = MagicMock()
    long_doc_content = CODE_SAMPLE_LONG_FUNCTION
    visitor.raw_documents = [
        Document(page_content=long_doc_content, metadata={"name": "long_function", "source": FILE_PATH})
    ]

    splitter = get_splitter(chunk_size=100, chunk_overlap=0) # Small chunk size for testing splitting

    final_docs = split_code_by_length(splitter, visitor)

    # The long function should be split into multiple smaller documents
    assert len(final_docs) > 1

    # Check that metadata is preserved
    for doc in final_docs:
        assert doc.metadata["name"] == "long_function"
        assert doc.metadata["source"] == FILE_PATH
    
    # Check that the total content length is roughly the same (allowing for minor splitter adjustments)
    combined_content = "".join([doc.page_content for doc in final_docs])
    # A robust check would be to ensure all lines from original are in combined,
    # but for now, checking relative length and presence of key parts is sufficient.
    assert len(combined_content) > len(long_doc_content) * 0.8 # Ensure substantial content is there
    assert "def long_function():" in combined_content
    assert "Line 100" in combined_content
    assert len(final_docs[0].page_content) <= 100 # First chunk should be <= chunk_size
