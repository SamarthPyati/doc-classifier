from langchain_huggingface import HuggingFaceEmbeddings
from config import DEFAULT_RAG_CONFIG, RAGConfig, EmbeddingProvider

import logging 
logger = logging.getLogger(__name__)

class Embeddings:
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config
        self.embedding_provider = self.config.Database.embedding_provider
        self.embedding_model = self.config.Database.embedding_model

    def _get_huggingface_model(self, device: str = 'mps', normalize_embeddings: bool = True) -> HuggingFaceEmbeddings: 
        try: 
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model, 
                model_kwargs={'device' : device},
                encode_kwargs={'normalize_embeddings': normalize_embeddings}
            )
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    def get_embedding_model(self): 
        match self.embedding_provider: 
            case EmbeddingProvider.HUGGINGFACE:
                return self._get_huggingface_model()
            case _: 
                logger.warning("Invalid embedding provider selected: Choose from: ", EmbeddingProvider._member_names_)