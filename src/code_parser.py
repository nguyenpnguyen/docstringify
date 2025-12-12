import ast
import os
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document

class CodeChunk(BaseModel):
    name: str              # name of code chunk (likely to be function or class name)
    type: str              # "function", "method", or "class"
    file_path: str
    start_line: int
    end_line: int
    content: str
    parent: Optional[str] = None
    
    @property
    def unique_id(self):
        """Creates a stable ID for the vector DB."""
        parent_prefix = f"{self.parent}." if self.parent else ""
        return f"{self.file_path}::{parent_prefix}{self.name}"

class PythonParser(ast.NodeVisitor):
    def __init__(self, source_code: str, file_path: str):
        self.source_code = source_code
        self.file_path = file_path
        self.chunks: List[CodeChunk] = []
        self.current_class: Optional[str] = None

    def _extract_code(self, node) -> str:
        """Extracts the exact source code string for a node."""
        return ast.get_source_segment(self.source_code, node) or ""

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visits a class definition."""
        # 1. Record the class itself as a chunk
        chunk = CodeChunk(
            name=node.name,
            type="class",
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            content=self._extract_code(node),
            parent=None
        )
        self.chunks.append(chunk)

        # 2. Dive deeper into the class (to find methods)
        # We track the 'current_class' so methods know who they belong to.
        previous_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node) # Continue recursion
        self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visits a function definition (sync)."""
        self._add_function_chunk(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visits a function definition (async)."""
        self._add_function_chunk(node)

    def _add_function_chunk(self, node):
        """Helper to create chunks for both sync and async functions."""
        # Determine if it's a standalone function or a method
        chunk_type = "method" if self.current_class else "function"
        
        chunk = CodeChunk(
            name=node.name,
            type=chunk_type,
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            content=self._extract_code(node),
            parent=self.current_class
        )
        self.chunks.append(chunk)

def parse_repository(repo_path: str) -> List[CodeChunk]:
    """
    Recursively scans a folder and parses all Python files.
    """
    all_chunks = []
    
    print(f"📂 Scanning repository: {repo_path}...")
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                
                # Skip virtual environments or hidden files
                if "venv" in full_path or "/." in full_path:
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    
                    # Parse the file
                    tree = ast.parse(source)
                    visitor = PythonParser(source, full_path)
                    visitor.visit(tree)
                    
                    all_chunks.extend(visitor.chunks)
                    
                except Exception as e:
                    print(f"⚠️ Error parsing {full_path}: {e}")

    print(f"✅ Found {len(all_chunks)} code chunks.")
    return all_chunks

def chunks_to_documents(chunks: List[CodeChunk]) -> List[Document]:
    return [
        Document(
            page_content=chunk.content,
            metadata={
                "source": chunk.file_path,
                "name": chunk.name,
                "type": chunk.type,
                "start_line": chunk.start_line
            }
        )
        for chunk in chunks
    ]
