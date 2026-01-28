import ast
import builtins
import logging

from typing import Optional
from langchain_text_splitters import TextSplitter, PythonCodeTextSplitter
from langchain_core.documents import Document

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class CodeChunkMetadata(BaseModel):
    path: str
    name: str
    type: str  # e.g., "function", "class_definition", "method"
    parent_class: Optional[str] = None  # For methods, the class they belong
    line_number: Optional[int] = None
    docstring: Optional[str] = None
    calls: list[str] = Field(default_factory=list) # store function calls in code chunk

class CodeStructureVisitor(ast.NodeVisitor):
    def __init__(self, source_code: str, file_path: str):
        self.source_code = source_code
        self.file_path = file_path
        self.raw_documents: list[Document] = []
        self.lines = source_code.splitlines(keepends=True)
        self.current_class: Optional[str] = None
        self.nodes_to_skip: set[ast.AST] = set()
        self.builtins_set = set(dir(builtins))
        self.aliases: dict[str, str] = {}  # to track imported aliases

    def _get_code_segment(self, node) -> str:
        return ast.get_source_segment(self.source_code, node) or ""
    def visit_Import(self, node):
        """
        Handles: import numpy as np
        """
        for alias in node.names:
            # name='numpy', asname='np'
            # If no asname, alias is the name itself (import math -> math)
            local_name = alias.asname or alias.name
            self.aliases[local_name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Handles: from utils import clean_text as clean
        """
        module = node.module or ""
        for alias in node.names:
            # alias.name = 'clean_text'
            # alias.asname = 'clean'
            full_name = f"{module}.{alias.name}" if module else alias.name
            local_name = alias.asname or alias.name
            self.aliases[local_name] = full_name
        self.generic_visit(node)

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """
        Helper to traverse a specific node (Function/Method) 
        and extract all function calls within it.
        """
        calls = set()
        # ast.walk recursively visits every child node
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # We use a mini-helper to resolve the name (e.g. 'self.db.save')
                func_name = self._resolve_name(child.func)
                if func_name and func_name not in self.builtins_set:
                    calls.add(func_name)
        return list(calls)

    def _resolve_name(self, node) -> Optional[str]:
        """Recursive helper to handle foo.bar.baz()"""
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            value = self._resolve_name(node.value)
            
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        return None

    def visit_FunctionDef(self, node):
        if node in self.nodes_to_skip:
            return
        self._add_function_doc(node, "function")

    def visit_AsyncFunctionDef(self, node):
        if node in self.nodes_to_skip:
            return
        self._add_function_doc(node, "async_function")

    def _add_function_doc(self, node, type_name):
        content = self._get_code_segment(node)
        docstring = ast.get_docstring(node)
        
        # Determine chunk type
        actual_type = "method" if self.current_class else type_name

        calls = self._extract_calls(node)

        metadata = CodeChunkMetadata(
            path=self.file_path,
            name=node.name,
            type=actual_type,
            parent_class=self.current_class,
            line_number=node.lineno,
            docstring=docstring,
            calls=calls
        )

        self.raw_documents.append(
            Document(page_content=content, metadata=metadata.model_dump())
        )

    def visit_ClassDef(self, node):
        split_node = None
        
        # Identify methods to handle __init__ merging
        methods = [
            child for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        # Initialize calls list for the class chunk
        class_calls = []

        if methods:
            first_method = methods[0]
            if first_method.name == "__init__":
                self.nodes_to_skip.add(first_method)
                
                # NEW: If we are merging __init__, we must capture its calls
                class_calls = self._extract_calls(first_method)

                if len(methods) > 1:
                    split_node = methods[1]
                else:
                    split_node = None
            else:
                split_node = first_method

        # Extract Code Segment
        if split_node:
            start_index = node.lineno - 1
            end_index = split_node.lineno - 1
            
            # Boundary checks
            if start_index < 0: start_index = 0
            if end_index > len(self.lines): end_index = len(self.lines)

            class_header = "".join(self.lines[start_index:end_index])
        else:
            class_header = self._get_code_segment(node)

        docstring = ast.get_docstring(node)

        metadata = CodeChunkMetadata(
            path=self.file_path,
            name=node.name,
            type="class_definition",
            line_number=node.lineno,
            docstring=docstring,
            calls=class_calls # Add __init__ calls here
        )

        self.raw_documents.append(
            Document(page_content=class_header, metadata=metadata.model_dump())
        )

        # Context Management
        previous_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = previous_class

def get_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    """Returns a Python-aware text splitter configuration."""
    return PythonCodeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def parse_code_structure(visitor: CodeStructureVisitor, code: str, file_path: str) -> None:
    """Parses code to extract logical structures using AST."""
    try:
        tree = ast.parse(code)
        visitor.visit(tree)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in file {file_path}: {e}")

def split_code_by_length(splitter: TextSplitter, visitor: CodeStructureVisitor) -> list[Document]:
    final_docs = []

    split_docs = splitter.split_documents(visitor.raw_documents)
    final_docs.extend(split_docs)

    return final_docs
