import os
import ast
from typing import List
from src.CodeStructureVisitor import CodeStructureVisitor
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

