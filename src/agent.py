from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loader import load_llm, load_embeddings, load_vector_store

LLM_ID = "qwen3:4b-instruct"
EMBED_ID = "qwen3-embedding:0.6b"
COLLECTION_NAME = "example_collection"
PERSIST_DIRECTORY = "./chroma_langchain_db"

llm: BaseChatModel = load_llm(model=LLM_ID, temperature=0.2, num_ctx=8192)

embeddings: Embeddings = load_embeddings(model=EMBED_ID)

vector_store: VectorStore = load_vector_store(
    embedder=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
)

if __name__ == "__main__":
    print(f'Embedding model: {embeddings.model}')
    print(f'LLM model: {llm.model}')
    print(llm.invoke("Hi, who are you?").content)

    embed_result = embeddings.embed_documents(["Hello world", "Hi there"])
    print(f'Embedding results: {embed_result}')
