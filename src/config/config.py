from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Tuple, Dict, List
import yaml

from src.constants import (
    EmbeddingProvider,
    EmbeddingModelHuggingFace,
    LLMModel,
    LLMModelProvider, 
    ChunkerType, 
    VectorStoreProvider, 
    ClassificationMethod
)

from pathlib import Path

def load_keywords_from_yaml(path: str = "keywords.yml") -> Dict[str, List[str]]:
    """ Loads classification keywords from a YAML file """
    try:
        # Resolve path relative to this file
        config_dir = Path(__file__).parent
        full_path = config_dir / path
        
        with open(full_path, 'r') as f:
            result: Dict[str, List[str]] = yaml.safe_load(f)
            return result
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load keywords from {path}. Error: {e}")
        return {"GENERAL": []}
    
class DocProcessorSettings(BaseModel):
    """ Settings for document processing and chunking """
    # Chunking strategy
    chunker_type: ChunkerType = ChunkerType.RECURSIVE_CHARACTER_TEXT_SPLITTER

    # Chunk size to split the document in (Makes the model get precise context to answer better)
    chunk_size: int = 2000
    chunk_overlap: int = 500

    # PDF specific settings
    pdf_extract_images: bool = False
    pdf_table_structure_infer_mode: Literal['csv', 'markdown', 'html', None] = 'csv'
    
    # Supported document file extensions
    # Office documents: PDF, Word, PowerPoint, Excel, RTF, OpenDocument
    # Web formats: HTML, XHTML, Markdown
    # Data formats: CSV, JSON, XML, YAML
    # E-books: EPUB
    # Plain text: TXT
    supported_extensions: Tuple[str, ...] = (
        ".pdf", ".txt", ".md", ".xhtml", ".html", ".docx",
        ".pptx", ".xlsx", ".xls", ".rtf", ".odt",
        ".csv", ".json", ".xml", ".yaml", ".yml",
        ".epub"
    )

    # Settings for classification 
    enable_classification: bool = True
    classification_method: ClassificationMethod = ClassificationMethod.KEYWORD
    classification_keywords: Dict[str, List[str]] = Field(default_factory=load_keywords_from_yaml)

    @property
    def classification_categories(self) -> List[str]: 
        return list(self.classification_keywords.keys())
    
class EmbeddingSettings(BaseModel):
    """ Settings for text embedding models and providers """
    provider: EmbeddingProvider = Field(default=EmbeddingProvider.GOOGLE, alias="embedding_provider")
    normalize: bool = Field(default=True, alias="normalize_embeddings")
    output_dimensionality: int | None = Field(default=768)

    # Model names for different providers
    huggingface_model: EmbeddingModelHuggingFace = Field(default=EmbeddingModelHuggingFace.MINI_LM, alias="embedding_model_huggingface")
    google_model: str = Field(default="models/text-embedding-004", alias="embedding_model_google")
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
    max_results: int = 10

    # NOTE: Set threshold to a lower value preferrably .2 or .3 if using a lightweight embedding model
    # Min threshold to classify as related in similarity search
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
     
    # TODO: Reranking

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