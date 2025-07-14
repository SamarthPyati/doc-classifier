from dataclasses import dataclass
from enum import Enum

class LLMModel(Enum):
    GEMMA: str = "gemma3:4b"
    LLAMA: str = "llama3.2:3b"


@dataclass
class RAGConfig: 
    """ Configuration for RAG """
    corpus_path: str = "corpus"
    llm_model: str = LLMModel.LLAMA
    
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
        max_results: int = 10

        # Min threshold to classify as related in similarity search
        similarity_threshold: float = .2

# TODO: Instead of a single config make multiple configs for each rag component
DEFAULT_RAG_CONFIG: RAGConfig = RAGConfig()

