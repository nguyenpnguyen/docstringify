import sqlite3, sqlite_vec
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loader import load_llm, load_embeddings, load_vector_store

LLM_ID = "qwen3:4b-instruct"
EMBED_ID = "qwen3-embedding:0.6b"
TABLE = "test_table"
DB_FILE = "test_vec.db"

llm: BaseChatModel = load_llm(model=LLM_ID, temperature=0.2, num_ctx=8192)

embeddings: Embeddings = load_embeddings(model=EMBED_ID)

db = sqlite3.connect(database=DB_FILE)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

vector_store: VectorStore = load_vector_store(embedding=embeddings, connection=db, table=TABLE, db_file=DB_FILE)

if __name__ == "__main__":
    print(f'Embedding model: {embeddings.model}')
    print(f'LLM model: {llm.model}')
    print(llm.invoke("Hi, who are you?").content)

    embed_result = embeddings.embed_documents(["Hello world", "Hi there"])
    print(f'Embedding results: {embed_result}')
