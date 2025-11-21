from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from .config import DEFAULT_RAG_CONFIG, RAGConfig
from .constants import EmbeddingProvider

import functools
import logging

logger = logging.getLogger(__name__)



class Embeddings:
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        self.config = config
        self.embedding_provider = self.config.Embeddings.provider
        self.__embedding_model_huggingface = self.config.Embeddings.huggingface_model.value
        self.__embedding_model_google = self.config.Embeddings.google_model
        self.__embedding_model_openai = self.config.Embeddings.openai_model
        
        self.output_dimensionality = self.config.Embeddings.output_dimensionality

    def _get_huggingface_model(self, device: str = 'mps', normalize_embeddings: bool = True) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.__embedding_model_huggingface,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )

    def _get_gemini_model(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=self.__embedding_model_google, 
            task_type='retrieval_document',  
        )

    def _get_openai_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.__embedding_model_openai
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
