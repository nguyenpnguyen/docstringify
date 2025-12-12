import logging
from src.loader import load_llm, load_embeddings, load_vector_store

LLM_ID = "qwen3:4b-instruct"
EMBED_ID = "qwen3-embedding:0.6b"
COLLECTION_NAME = "test_collection"
PERSIST_DIRECTORY = "./test_chroma_db"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def test_loader():
    assert load_llm(model=LLM_ID) is not None, logger.error("LLM loading failed")
    logger.info("LLM loaded successfully")

    embedder = load_embeddings(model=EMBED_ID)
    assert embedder is not None, logger.error("Embeddings loading failed")
    logger.info("Embeddings loaded successfully")

    assert load_vector_store(embedder=embedder,
                             collection_name=COLLECTION_NAME,
                             persist_directory=PERSIST_DIRECTORY
                            ) is not None, logger.error("Vector store loading failed")

    logger.info("Vector store loaded successfully")
