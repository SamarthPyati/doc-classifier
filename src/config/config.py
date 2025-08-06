from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Tuple

from src.constants import (
    EmbeddingProvider,
    EmbeddingModelHuggingFace,
    LLMModel,
    LLMModelProvider, 
    ChunkerType, 
    VectorStoreProvider
)

class DocProcessorSettings(BaseModel):
    """ Settings for document processing and chunking """
    # Chunking strategy
    chunker_type: ChunkerType = ChunkerType.RECURSIVE_CHARACTER_TEXT_SPLITTER

    # Chunk size to split the document in (Makes the model get precise context to answer better)
    chunk_size: int = 400
    chunk_overlap: int = 0
    
    # PDF specific settings
    pdf_extract_images: bool = False
    pdf_table_structure_infer_mode: Literal['csv', 'markdown', 'html', None] = 'csv'
    
    # TODO: Add more document types for support
    supported_extensions: Tuple[str, ...] = (".pdf", ".txt", ".md", ".xhtml", ".html", ".docx")

    # Settings for classification 
    enable_classification: bool = True
    classification_categories: Tuple[str, ...] = ("HR", "Procurement", "Legal", "Finance", "Technical", "General")
    
    # Load documents lazily into a iterator
    lazy_loading: bool = True

class EmbeddingSettings(BaseModel):
    """ Settings for text embedding models and providers """
    provider: EmbeddingProvider = Field(default=EmbeddingProvider.GOOGLE, alias="embedding_provider")
    normalize: bool = Field(default=True, alias="normalize_embeddings")

    # Model names for different providers
    huggingface_model: EmbeddingModelHuggingFace = Field(default=EmbeddingModelHuggingFace.MINI_LM, alias="embedding_model_huggingface")
    google_model: str = Field(default="models/embedding-001", alias="embedding_model_google")
    openai_model: str = Field(default="text-embedding-3-small", alias="embedding_model_openai")

    # Batch operations 
    batch_size: int = 1000
class DatabaseSettings(BaseModel):
    """ Settings for the vector database """
    provider: VectorStoreProvider = VectorStoreProvider.CHROMA

    # Chroma db path to store locally
    path: str = Field(default="chroma_db", alias="database_path")
    collection_name: str = "rag-index"

    # Max results to return after similarity search
    max_results: int = 5

    # NOTE: Set threshold to a lower value preferrably .2 or .3 if using a lightweight embedding model
    # Min threshold to classify as related in similarity search
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
     
    # TODO: Reranking
    enable_reranking: bool = True
    rerank_top_k: int = 20

class LLMSettings(BaseModel):
    """ Settings for the Large Language Model which responds to chat and query """
    model: LLMModel = Field(default=LLMModel.GEMINI_FLASH_LITE_2_0, alias="llm_model")
    provider: LLMModelProvider = Field(default=LLMModelProvider.GOOGLE, alias="llm_provider")

    # LLM Hyperparams
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # Settings for local models via Ollama
    ctx_window_size: int = 4096
    num_threads: int = 4

    # Performance settings
    streaming: bool = True
    batch_inference: bool = True
    max_batch_size: int = 8

class RAGConfig(BaseSettings):
    """
    Main configuration for the RAG system, loaded from environment variables or defaults.
    """

    # Configure Pydantic to load from a .env file, use a prefix, and handle nested structures.
    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='RAG_',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore'
    )

    corpus_path: str = "corpus"

    DocProcessor: DocProcessorSettings = DocProcessorSettings()
    Embeddings: EmbeddingSettings = EmbeddingSettings()
    Database: DatabaseSettings = DatabaseSettings()
    LLM: LLMSettings = LLMSettings()

DEFAULT_RAG_CONFIG: RAGConfig = RAGConfig()