from src.code_parser import get_splitter, load_and_split_repository
from pathlib import Path

CODE_SAMPLE = """
def hello_world():
    print('Hello, world!')
"""

SAMPLE_REPO_PATH = "tests/sample"

splitter = get_splitter()


def test_load_and_split_repository_empty():
    repo_path = SAMPLE_REPO_PATH

    files = []
    for file in Path(repo_path).rglob("*.py"):
        files.append(file)

    docs = load_and_split_repository(repo_path)
    assert len(docs) > 0, "No documents were created from the sample repository."
    for i, doc in enumerate(docs):
        assert "source" in doc.metadata, "Document metadata missing 'source' key."
        assert doc.metadata["source"].endswith(".py"), "Document source is not a Python file."
        assert doc.metadata["source"] in [str(f) for f in files], "Document source not found in the repository files."
