import os
from typing import List, Optional
from langchain_text_splitters import TextSplitter, PythonCodeTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

CHUNK_SIZE = 1500 
CHUNK_OVERLAP = 300 

def get_splitter() -> TextSplitter:
    """Returns a Python-aware text splitter configuration."""
    return PythonCodeTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

def split_code(code: str, splitter: TextSplitter, metadata: Optional[dict] = None) -> List[Document]:
    """
    Splits a raw code string into a list of Document objects with metadata.
    """
    # create_documents handles the splitting of text into chunks
    # and assigns the same metadata dictionary to each chunk created from this text.
    return splitter.create_documents([code], metadatas=[metadata] if metadata else None)

def load_and_split_repository(repo_path: str) -> List[Document]:
    """
    Scans a repository, loads Python files, and splits them into chunks
    using the defined splitter and settings.
    """
    all_docs = []
    splitter = get_splitter()

    print(f"📂 Scanning repository: {repo_path}...")

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

                    # Create documents with metadata
                    file_docs = split_code(
                        file_content, 
                        splitter, 
                        metadata={"source": full_path}
                    )
                    
                    all_docs.extend(file_docs)

                except Exception as e:
                    print(f"⚠️ Error processing {full_path}: {e}")

    print(f"✅ processed {len(all_docs)} code chunks.")
    return all_docs

def save_documents_to_vector_store(documents: List[Document], vector_store: VectorStore) -> List[str]:
    """
    Saves the list of Document objects to the provided vector store.
    """
    return vector_store.add_documents(documents)
