import logging
from src.loader import load_llm, load_embeddings, load_vector_store
from langchain_chroma import Chroma

# --- Configuration ---
LLM_ID = "qwen3:4b-instruct"
EMBED_ID = "qwen3-embedding:0.6b"
TEST_COLLECTION_NAME = "test_loader_collection"

# --- Logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Tests ---

def test_model_loaders():
    """
    Tests the successful loading of the LLM and embedding models.
    """
    # Test LLM loading
    llm = load_llm(model=LLM_ID)
    assert llm is not None, "LLM loading failed"
    logger.info("LLM loaded successfully")

    # Test Embeddings loading
    embedding = load_embeddings(model=EMBED_ID)
    assert embedding is not None, "Embeddings loading failed"
    logger.info("Embeddings loaded successfully")


def test_vector_store_loader():
    """
    Tests the successful loading of the Chroma vector store.
    """
    # Load embeddings required by the vector store
    embedding_model = load_embeddings(model=EMBED_ID)

    # Call the function to load the vector store
    vector_store = load_vector_store(
        collection_name=TEST_COLLECTION_NAME,
        embedding=embedding_model
    )

    # 1. Assert that the vector store object was created
    assert vector_store is not None, "Vector store loading failed"

    # 2. Assert that the created object is a Chroma instance
    assert isinstance(vector_store, Chroma), "Vector store is not a Chroma instance."
    logger.info("Chroma vector store loaded successfully")

    # 3. Assert the collection name is set correctly
    assert vector_store._collection.name == TEST_COLLECTION_NAME
