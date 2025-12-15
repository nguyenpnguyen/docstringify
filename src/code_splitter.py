import os
import ast
from typing import List, Optional, Set
from langchain_text_splitters import TextSplitter, PythonCodeTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def get_splitter() -> TextSplitter:
    """Returns a Python-aware text splitter configuration."""
    return PythonCodeTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

class CodeStructureVisitor(ast.NodeVisitor):
    """
    Extracts class and function definitions from source code
    to create semantically meaningful chunks.
    """
    def __init__(self, source_code: str, file_path: str):
        self.source_code = source_code
        self.file_path = file_path
        self.raw_documents: List[Document] = []
        self.lines = source_code.splitlines(keepends=True)
        # Track context
        self.current_class: Optional[str] = None
        self.nodes_to_skip: Set[ast.AST] = set()

    def _get_code_segment(self, node) -> str:
        return ast.get_source_segment(self.source_code, node) or ""

    def visit_FunctionDef(self, node):
        if node in self.nodes_to_skip:
            return
        self._add_function_doc(node, "function")

    def visit_AsyncFunctionDef(self, node):
        if node in self.nodes_to_skip:
            return
        self._add_function_doc(node, "async_function")

    def _add_function_doc(self, node, type_name):
        """Creates a Document for a function."""
        content = self._get_code_segment(node)
        # Extract existing docstring (None if missing)
        docstring = ast.get_docstring(node)
        
        # Determine chunk type (method vs function) based on context
        actual_type = "method" if self.current_class else type_name

        self.raw_documents.append(Document(
            page_content=content,
            metadata={
                "source": self.file_path,
                "name": node.name,
                "type": actual_type,
                "parent_class": self.current_class, # Context awareness
                "start_line": node.lineno,
                "docstring": docstring
            }
        ))
        # We do not visit children to avoid duplicating nested functions
        # unless you specifically want closures indexed separately.

    def visit_ClassDef(self, node):
        """
        Captures class definition. 
        Includes __init__ method if it is the first method.
        Stops before the first non-init method.
        """
        # 1. Analyze children to find split point
        split_node = None
        
        # We iterate to find the first method and see if it is __init__
        methods = [child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        
        if methods:
            first_method = methods[0]
            if first_method.name == "__init__":
                # We want to INCLUDE __init__ in the class chunk
                # So we skip creating a separate chunk for it
                self.nodes_to_skip.add(first_method)
                
                # The cutoff is the *next* method if it exists
                if len(methods) > 1:
                    split_node = methods[1]
                else:
                    # __init__ is the only method, take the whole class
                    split_node = None 
            else:
                # First method is NOT __init__, so we stop before it
                split_node = first_method
        
        # 2. Extract code
        if split_node:
            # node.lineno is 1-based start of class
            # split_node.lineno is 1-based start of the next method
            start_index = node.lineno - 1
            end_index = split_node.lineno - 1
            
            # Safety check
            if start_index < 0: start_index = 0
            if end_index > len(self.lines): end_index = len(self.lines)
            
            class_header = "".join(self.lines[start_index:end_index])
        else:
            # Take the whole thing (either no methods, or only __init__)
            class_header = self._get_code_segment(node)

        docstring = ast.get_docstring(node)

        self.raw_documents.append(Document(
            page_content=class_header,
            metadata={
                "source": self.file_path,
                "name": node.name,
                "type": "class_definition",
                "start_line": node.lineno,
                "docstring": docstring
            }
        ))

        # 3. Manage Context and Recurse
        previous_class = self.current_class
        self.current_class = node.name
        
        self.generic_visit(node)
        
        self.current_class = previous_class


def load_and_split_repository(repo_path: str) -> List[Document]:
    """
    Scans a repository, parses AST to find logical blocks,
    and then splits them if they exceed chunk limits.
    """
    final_docs = []
    splitter = get_splitter()

    print(f"Scanning repository: {repo_path}...")

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                
                # Skip virtual envs and hidden folders
                if "venv" in full_path or "/." in full_path:
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        file_content = f.read()

                    # 1. Parse Structure (AST Phase)
                    # This creates "Logical Chunks" (Whole functions/classes)
                    visitor = CodeStructureVisitor(file_content, full_path)
                    try:
                        tree = ast.parse(file_content)
                        visitor.visit(tree)
                    except SyntaxError:
                        print(f"Syntax Error parsing {full_path}, skipping AST.")
                        continue

                    # 2. Text Split (Size Phase)
                    # If a function is > 1500 chars, this splits it.
                    # Metadata is preserved from the AST phase!
                    if visitor.raw_documents:
                        split_docs = splitter.split_documents(visitor.raw_documents)
                        final_docs.extend(split_docs)

                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

    print(f"processed {len(final_docs)} code chunks.")
    return final_docs
