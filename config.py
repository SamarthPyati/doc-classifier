from dataclasses import dataclass
from enum import Enum
import logging

# Logger Config
logging.basicConfig(
    filename="events.log", 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMModel(Enum):
    GEMMA: str = "gemma3:4b"
    LLAMA: str = "llama3.2:3b"

@dataclass
class RAGConfig: 
    """ Configuration for RAG """
    corpus_path: str = "corpus"
    llm_model: str = LLMModel.LLAMA.value
    
    class DocProcessor: 
        # Text Splitter chunk size and chunk conflict
        chunk_size: int = 1000
        chunk_overlap: int = 250

    class Database: 
        # Chroma db path to store locally
        database_path: str = "chroma"
        
        # Embedding model 
        embedding_model: str = "all-MiniLM-L6-v2"
        
        # Max results to return after similarity search 
        max_results: int = 5

        # Min threshold to classify as related in similarity search
        similarity_threshold: float = .5

    class LLM:
        # LLM specific settings
        temperature: float = 0.1
        max_tokens: int = 2048
        top_p: float = 0.9

# TODO: Instead of a single config make multiple configs for each rag component
DEFAULT_RAG_CONFIG: RAGConfig = RAGConfig()

