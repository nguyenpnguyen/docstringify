import os
import logging

from langchain_core.documents import Document

from src.code_parser import CodeStructureVisitor, parse_code_structure, split_code_by_length, get_splitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def load_and_split_repository(repo_path: str) -> list[Document]:
    """
    Scans a repository, parses AST to find logical blocks,
    and then splits them if they exceed chunk limits.
    """
    final_docs = []
    splitter = get_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    logger.debug(f"Scanning repository: {repo_path}...")

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
                        parse_code_structure(visitor, file_content, full_path)
                    except SyntaxError:
                        logger.error(f"Syntax Error parsing {full_path}, skipping AST.")
                        continue

                    # 2. Text Split (Size Phase)
                    # If a function is > 1500 chars, this splits it.
                    # Metadata is preserved from the AST phase!
                    if visitor.raw_documents:
                        final_docs.extend(split_code_by_length(splitter, visitor))

                except Exception as e:
                    logger.error(f"Error processing {full_path}: {e}")

    logger.debug(f"processed {len(final_docs)} code chunks.")

    return final_docs

def index_codebase(documents: list[Document]):
    pass
