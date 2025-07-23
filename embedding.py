from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import DEFAULT_RAG_CONFIG, RAGConfig, EmbeddingProvider

import logging
logger = logging.getLogger(__name__)

class Embeddings:
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        self.config = config
        self.embedding_provider = self.config.Database.embedding_provider
        self.embedding_model = self.config.Database.embedding_model_hugging_face.value

    def _get_huggingface_model(self, device: str = 'mps', normalize_embeddings: bool = True) -> HuggingFaceEmbeddings:
        try:
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': normalize_embeddings}
            )
        except Exception as e:
            logger.error(f"Error initializing HuggingFace embedding model: {e}")
            raise

    def _get_gemini_model(self) -> GoogleGenerativeAIEmbeddings:
        try:
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
        except Exception as e:
            logger.error(f"Error initializing Google Generative AI embedding model: {e}")
            raise

    def get_embedding_model(self):
        match self.embedding_provider:
            case EmbeddingProvider.HUGGINGFACE:
                return self._get_huggingface_model()
            case EmbeddingProvider.GOOGLE:
                return self._get_gemini_model()
            case _:
                logger.warning("Invalid embedding provider selected. Choose from: %s",
                               ', '.join(EmbeddingProvider._member_names_))
                return None
