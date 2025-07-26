import logging
import yaml
import sys

from dataclasses import dataclass
from enum import Enum
from typing import Literal

# Logger Config (file and stdout logger)
file_handler = logging.FileHandler(filename='events.log', )
file_handler.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setFormatter(logging.Formatter("%(filename)s:%(lineno)d:%(funcName)s - %(levelname)s - %(message)s"))
stdout_handler.setLevel(logging.INFO)

logging.basicConfig(
    format="[%(asctime)s] %(module)s - %(levelname)s - %(message)s", 
    datefmt="%d-%m-%Y %H:%M:%S", 
    handlers=[file_handler, stdout_handler],
    level=logging.INFO
)

logger = logging.getLogger(__name__)


class LLMModel(Enum):
    # Ollama models (run locally)
    GEMMA = "gemma3:4b"
    LLAMA = "llama3.2:3b"

    # Google gemini models 
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-2.5-pro"

class LLMModelProvider(Enum):
    GOOGLE = "google_genai"

class EmbeddingProvider(Enum):
    HUGGINGFACE = "HuggingFace"
    GOOGLE = "Google-Gemini"
    OPENAI = "Openai"

class EmbeddingModelHuggingFace(Enum):
    MINI_LM = "all-MiniLM-L6-v2"            # Embedding length - 364
    LABSE   = "LaBSE"                       # Embedding length - 728
    ROBERTA = "all-roberta-large-v1"        # Embedding length - 1024

@dataclass
class RAGConfig: 
    """ Configuration for RAG """
    corpus_path: str = "corpus"
    
    @dataclass
    class DocProcessor: 
        # Takes long time 
        use_semantic_chunking: bool = False

        # Chunk size to split the document in (Makes the model get precise context to answer better)
        chunk_size: int = 1000
        
        # Overlap token with adjacent chunks
        chunk_overlap: int = 200

        # PDF specific settings
        pdf_extract_images: bool = False    # Performance
        pdf_table_structure_infer_mode: Literal['csv', 'markdown', 'html', None] = 'csv'

        # Supported file extensions
        supported_extensions: tuple = (".pdf", ".txt", ".md", ".xhtml", ".html", ".docx")

        # Load documents lazily into a iterator
        lazy_loading: bool = True
    
    @dataclass
    class Embedding: 
        # Select the embedding provider from Hugginface or Google  
        embedding_provider: EmbeddingProvider = EmbeddingProvider.GOOGLE

        # HuggingFace Embedding models (set embedding_provider to HUGGINGFACE) 
        # ["all-MiniLM-L6-v2" (384), "LaBSE" (768), "all-roberta-large-v1" (1024)]
        embedding_model_huggingface: EmbeddingModelHuggingFace = EmbeddingModelHuggingFace.MINI_LM
        embedding_model_google: str = "models/embedding-001"
        embedding_model_openai: str = "text-embedding-3-small"
        normalize_embeddings: bool = True

        # Batch operations
        batch_size: int = 1000

    @dataclass
    class Database: 
        # Chroma db path to store locally
        database_path: str = "chroma_db"
        
        collection_name: str = "rag_documents"
        
        # Max results to return after similarity search 
        max_results: int = 10

        # NOTE: Set threshold to a lower value preferrably .2 or .3 if using a lightweight embedding model 
        # Min threshold to classify as related in similarity search
        similarity_threshold: float = 0.7

        # Reranking
        enable_reranking: bool = True
        rerank_top_k: int = 20


    @dataclass
    class LLM:
        llm_model: LLMModel = LLMModel.GEMINI_FLASH
        llm_provider: LLMModelProvider = LLMModelProvider.GOOGLE

        # LLM Hyperparams 
        temperature: float = 0.1
        top_p: float = 0.9

        # LLM Hyperparams (for local running models via ollama)
        ctx_window_size: int = 4096
        num_threads: int = 4

        # Performance settings
        streaming: bool = True
        batch_inference: bool = True
        max_batch_size: int = 8

    @classmethod
    def load_config_from_yaml(cls, file_path: str): 
        """ Load configuration externally from a yaml file """
        with open(file_path, 'r') as f: 
            config = yaml.safe_load(f)
        return cls(**config)

DEFAULT_RAG_CONFIG: RAGConfig = RAGConfig()

# TODO: Validation with Pydantic BaseSettings
# TODO: Make separate classes for each config rather than monolithic approach