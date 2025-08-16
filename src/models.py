from dataclasses import dataclass, field
from typing import List
from langchain.schema import Document

@dataclass 
class RAGContext: 
    """ Model to hold the context retrieved from the vector store """
    documents: List[Document] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0

@dataclass 
class Result:
    """ Model for the final output of a query or chat interaction """
    response: str       = "Null"    
    sources: List[str]  = field(default_factory=list)
    confidence: float   = 0.0
    num_sources: int    = 0
    processing_time: float = 0.0
    session_id: str = "Null"

    def __str__(self):
        return (f"\n🤖 Response: {self.response}\n"
                f"📚 Sources: {', '.join(self.sources)}\n"
                f"🎯 Confidence: {self.confidence:.3f}\n"
                f"⚡ Processing Time: {self.processing_time:.3f} second(s)\n")

    def __repr__(self):
        return (f"Response: {self.response}\n"
                f"Session ID: {self.session_id}\n"
                f"Sources: {self.sources}\n"
                f"Confidence: {self.confidence:.3f}\n"
                f"Number of sources: {self.num_sources}\n"
                f"Processing Time: {self.processing_time:.3f} second(s)\n")
