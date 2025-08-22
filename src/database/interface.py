from langchain.schema import Document

from src.config import RAGConfig, DEFAULT_RAG_CONFIG
from src.embedding import Embeddings

from abc import ABC, abstractmethod
from typing import List, Tuple

import logging 
logger = logging.getLogger(__name__)

class VectorStoreInterface(ABC):
    """ Abstract base class for vector store managers """
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config
        self.embedding_function = Embeddings(config).get_embedding_model()    
        self._db = self._initialize_db()

    @abstractmethod
    def _initialize_db(self): 
        """Initialize or create the DB if not existing."""
        pass

    @abstractmethod
    async def add_documents(self, chunks: List[Document], force_rebuild: bool = False) -> bool:
        """Adds documents to the vector store."""
        pass

    def similarity_search(self, query: str) -> List[Tuple[Document, float]]:
        """Performs a similarity search."""
        try:
            # Perform similarity search
            results = self._db.similarity_search_with_relevance_scores(
                query, 
                k=self.config.Database.max_results,  
                score_threshold=self.config.Database.similarity_threshold
            )
            # Sort by relevance score
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(results)} relevant documents for query: \"{query[:50]}...\"")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}", exc_info=True)
            return []

    def count(self) -> int: 
        """Returns the count of docs in the vector store."""
        return self._db._collection.count() if self._db else -1

    def peek(self, n: int): 
        """ Helper function to peek into database, just for testing purposes """
        docs = self._db._collection.peek(n)["metadatas"]
        for doc in docs:
            print(doc.get("source", "N/A"), " -> ", doc.get("file_category", "N/A"))

    def reset(self) -> bool:
        """ Reset/clear the entire database """
        try:
            if self._db is not None:
                self._db.reset_collection()
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False