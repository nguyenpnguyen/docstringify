from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma


def load_llm(model: str, temperature: float = 0.2, num_ctx: int = 8192) -> BaseChatModel:
    """
    Load LLM
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        validate_model_on_init=True,
        num_ctx=num_ctx,
    )


def load_embeddings(model: str) -> Embeddings:
    """
    Load embedding
    """
    return OllamaEmbeddings(
        model=model,
        validate_model_on_init=True,
    )


def load_vector_store(collection_name: str, embedding: Embeddings, **kwargs) -> VectorStore:
    return Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
    )
