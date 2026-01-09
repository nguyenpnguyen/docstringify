import logging

from typing import Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_llm(model: str, temperature: float = 0.2, num_ctx: int = 8192) -> Optional[BaseChatModel]:
    """
    Load LLM
    """
    try:
        return ChatOllama(
            model=model,
            temperature=temperature,
            validate_model_on_init=True,
            num_ctx=num_ctx,
        )
    except Exception as e:
        logger.error(f"Error loading LLM '{model}': {e}")


def load_embeddings(model: str) -> Optional[Embeddings]:
    """
    Load embedding
    """
    try:
        return OllamaEmbeddings(
            model=model,
            validate_model_on_init=True,
        )
    except Exception as e:
        logger.error(f"Error loading Embeddings model '{model}': {e}")

