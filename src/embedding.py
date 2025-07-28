from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from .config import DEFAULT_RAG_CONFIG, RAGConfig, EmbeddingProvider

import functools
import logging
logger = logging.getLogger(__name__)

def handle_embedding_errors(func): 
    """ Wrapper to handle the errors while initializing the embedding model """
    @functools.wraps(func)
    def wrapper(*args, **kwargs): 
        try: 
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error initializing embedding model in {func.__name__}: {e}")
            return None
    return wrapper

class Embeddings:
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        self.config = config
        self.embedding_provider = self.config.Embeddings.provider
        self.embedding_model_huggingface = self.config.Embeddings.huggingface_model.value
        self.embedding_model_google = self.config.Embeddings.google_model
        self.embedding_model_openai = self.config.Embeddings.openai_model

    @handle_embedding_errors
    def _get_huggingface_model(self, device: str = 'mps', normalize_embeddings: bool = True) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_huggingface,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )

    @handle_embedding_errors
    def _get_gemini_model(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=self.embedding_model_google, 
        )

    @handle_embedding_errors
    def _get_openai_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.embedding_model_openai
        )

    def get_embedding_model(self):
        match self.embedding_provider:
            case EmbeddingProvider.HUGGINGFACE:
                return self._get_huggingface_model()
            case EmbeddingProvider.GOOGLE:
                return self._get_gemini_model()
            case EmbeddingProvider.OPENAI: 
                return self._get_openai_model()
            case _:
                logger.warning("Invalid embedding provider selected. Choose from: %s",
                               ', '.join(EmbeddingProvider._member_names_))
                return None
