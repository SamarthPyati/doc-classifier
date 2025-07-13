from dataclasses import dataclass


@dataclass
class RAGConfig: 
    """ Configuration for RAG """
    
    corpus_path: str = "corpus"
    ollama_model: str = "gemma3:4b"
    
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
        similarity_threshold: float = .6

# TODO: Instead of a single config make multiple configs for each rag component
DEFAULT_RAG_CONFIG: RAGConfig = RAGConfig()

