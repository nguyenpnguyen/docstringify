import logging
import sqlite3
import sqlite_vec
import os
from src.loader import load_llm, load_embeddings, load_vector_store

LLM_ID = "qwen3:4b-instruct"
EMBED_ID = "qwen3-embedding:0.6b"
TABLE = "test_table"
DB_FILE = "/tmp/test_vec.db"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

embedding = load_embeddings(model=EMBED_ID)


def test_embedding_loader():
    assert load_llm(model=LLM_ID) is not None, logger.error("LLM loading failed")
    logger.info("LLM loaded successfully")

    assert embedding is not None, logger.error("Embeddings loading failed")
    logger.info("Embeddings loaded successfully")


def test_vector_store_loader():
    db = sqlite3.connect(database=DB_FILE)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    vector_store = load_vector_store(
        table=TABLE,
        connection=db,
        embedding=embedding,
        db_file=DB_FILE
    )

    assert vector_store is not None, logger.error("Vector store loading failed")
    logger.info("Vector store loaded successfully")

    os.remove(DB_FILE)
