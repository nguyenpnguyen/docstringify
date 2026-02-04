import json
import logging

from langchain_core.documents import Document
from src.db import CodeChunk, select_code_chunk_by_name, get_dependencies, get_dependents


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def chunk_to_document(chunk: CodeChunk) -> Document:
    return Document(
        page_content=chunk.content,
        metadata={
            "path": chunk.path,
            "name": chunk.name,
            "type": chunk.type,
            "parent_class": chunk.parent_class,
            "line_number": chunk.line_number,
            "docstring": chunk.docstring,
        },
    )

def dependency_retriever(name: str, path: str) -> list[Document]:
    chunks = get_dependencies(select_code_chunk_by_name(name, path))
    return [chunk_to_document(chunk) for chunk in chunks]


def usage_retriever(name: str, path: str) -> list[Document]:
    chunks = get_dependents(select_code_chunk_by_name(name, path))
    return [chunk_to_document(chunk) for chunk in chunks]

def rank_context(target_chunk: CodeChunk, context_chunks, top_n=10, dependency_ratio=0.6) -> list[CodeChunk]:
    """
    Sorts a mixed list of chunks for optimal context injection.
    """
    dependencies = []
    dependents = []
    
    # 1. Split into two buckets
    calls = json.loads(target_chunk.calls)
    for chunk in context_chunks:
        if chunk.name in calls:
            dependencies.append(chunk)
        else:
            dependents.append(chunk)

    # 2. Sort Dependencies (The "Internal Logic")
    # Criteria: Has Docstring > Same File > Alphabetical
    dependencies.sort(key=lambda x: (
        x.docstring is not None,         # Primary: Docs exist?
        x.path == target_chunk.path, # Secondary: Same file?
        x.name                           # Tertiary: Stable sort
    ), reverse=True)

    # 3. Sort Dependents (The "Usage Examples")
    # Criteria: Is Test > Different File > Alphabetical
    dependents.sort(key=lambda x: (
        "test" in x.path.lower(),      # Primary: Is it a test?
        x.path != target_chunk.path, # Secondary: External usage?
        x.name
    ), reverse=True)

    # 4. Select Top N
    num_dependencies = int(top_n * dependency_ratio)
    selected_deps = dependencies[:num_dependencies]
    selected_usages = dependents[:top_n - num_dependencies]
    
    # 5. Return ordered for Prompt Construction
    # Order: Usages -> Dependencies
    return selected_usages + selected_deps

def retrieve_relevant_docs(name: str, path: str) -> list[Document]:
    """
    Retrieves documents that are either dependencies or dependents
    of the specified code chunk.
    """
    try:
        chunk = select_code_chunk_by_name(name, path)
    except CodeChunk.DoesNotExist:
        logger.warning(f"CodeChunk with name '{name}' does not exist.")
        return []

    dependencies = get_dependencies(chunk)
    dependents = get_dependents(chunk)
    relevant_chunks = set(dependencies + dependents)
    context = rank_context(chunk, relevant_chunks, top_n=10, dependency_ratio=0.6)
    return [chunk_to_document(c) for c in context]
