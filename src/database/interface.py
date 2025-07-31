from langchain.schema import Document

from abc import ABC, abstractmethod
from typing import List, Tuple

class VectorStoreInterface(ABC):
    """ Abstract base class for vector store managers """
    @abstractmethod
    def add_documents(self, chunks: List[Document], force_rebuild: bool = False) -> bool:
        """Adds documents to the vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str) -> List[Tuple[Document, float]]:
        """Performs a similarity search."""
        pass

    @abstractmethod
    def count(self) -> int: 
        """Returns the count of docs in the vector store."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Resets the entire vector store."""
        pass