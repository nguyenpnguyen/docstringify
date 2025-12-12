from src.ingest import get_splitter, split_code, load_and_split_repository

CODE_SAMPLE = """
def hello_world():
    print('Hello, world!')
"""

SAMPLE_REPO_PATH = "./tests/gpt_oss_sample"

def test_ingest():
    splitter = get_splitter()

    # Test split_code
    code = CODE_SAMPLE
    docs = split_code(code, splitter, metadata={})
    assert len(docs) > 0, "No documents were created from the code sample."

    # Test load_and_split_repository
    repo_path = SAMPLE_REPO_PATH
    docs = load_and_split_repository(repo_path)
    assert len(docs) > 0, "No documents were created from the sample repository."
    for doc in docs:
        assert "source" in doc.metadata, "Document metadata missing 'source' key."
        assert doc.metadata["source"].endswith(".py"), "Document source is not a Python file."
