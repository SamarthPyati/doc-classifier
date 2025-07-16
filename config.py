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
        # Chunk size to split the document in (Makes the model get precise context to answer better)
        chunk_size: int = 2000
        
        # Overlap token with adjancent chunks
        chunk_overlap: int = 250

    class Database: 
        # Chroma db path to store locally
        database_path: str = "chroma"
        
        # Embedding models (Should be belonging to sentence_transformers) = ["all-MiniLM-L6-v2", "LaBSE", "all-roberta-large-v1"]
        embedding_model: str = "all-roberta-large-v1"
        
        # Max results to return after similarity search 
        max_results: int = 10

        # Min threshold to classify as related in similarity search
        # NOTE: If a better embedding model is used, increase this to provide much better results, else 
        #       if a tiny model is used lowering would the threshold to about .2 ~ .3 would be good
        similarity_threshold: float = .2

    class LLM:
        temperature: float = 0.1
        ctx_window_size: int = 2048
        top_p: float = 0.9

# TODO: Instead of a single config make multiple configs for each rag component
DEFAULT_RAG_CONFIG: RAGConfig = RAGConfig()

