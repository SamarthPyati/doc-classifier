from langchain_ollama import OllamaLLM

from document import DocumentProcessor, CosineEmbeddingEvaluator
from database import VectorStoreManager

from config import RAGConfig, DEFAULT_RAG_CONFIG, logger

from typing import Dict, Any, List
from dataclasses import dataclass, field

@dataclass 
class QueryResult:
    response: str       = "No answer generated"
    sources: List[str]  = field(default_factory=list)
    confidence: float   = 0.0
    num_sources: int    = 0

    def __repr__(self):
        return (f"Response: {self.response}\n"
                f"Sources: {self.sources}\n"
                f"Confidence: {self.confidence:.3f}\n"
                f"Number of sources: {self.num_sources}")

class RAGSystem: 
    """ Main RAG System orchestrator """
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):     
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        self.prompt_template = """
            Answer the question based only on the following context:
    
            {context}
    
            ======================================================================
    
            Answer the question based on the above context: {question}
        """ 
        self.llm = self._initialize_llm()

        # Load vector store
        self.db = None

    def _initialize_llm(self): 
        try: 
            # Configure OllamaLLM
            return OllamaLLM(
                model=self.config.llm_model, 
                num_thread=8, 
                temperature=self.config.LLM.temperature, 
                num_ctx=self.config.LLM.ctx_window_size, 
                top_p=self.config.LLM.top_p, 
            )
        except Exception as e: 
            logger.error(f"Error initializing llm: {e}")
            raise

    # TODO: Automatically detect new document upload and rebuild the knowledge base
    def build_knowledge_base(self) -> None:
        """ Build the knowledge base from documents """
        logger.info("Building knowledge base...")
        
        # Load and process documents
        documents = self.document_processor.load_documents()
        if not documents:
            raise ValueError("No documents found to process")
        
        # Split documents into chunks
        chunks = self.document_processor.split_documents(documents)
        
        # Create vector store
        self.vector_store_manager.create_vector_store(chunks)
        
        logger.info("Knowledge base built successfully")

    def query(self, question: str) -> QueryResult:
        """ Query the RAG system with a question """
        try:
            self.db = self.vector_store_manager.load_vector_store()
            
            # Perform similarity search
            results = self.vector_store_manager.similarity_search(question, self.db)
            
            if not results:
                return QueryResult(response="No relevant documents found")
            
            # Prepare context
            context = "\n\n======================================================================\n\n".join([
                doc.page_content for doc, _score in results
            ])
            
            # Generate prompt
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Get LLM response
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
            else:
                response = self.llm.predict(prompt).content
            
            # Extract sources
            sources = list(set([
                doc.metadata.get("source", "Unknown") for doc, _score in results
            ]))
            
            # Calculate average confidence
            avg_confidence = sum(score for _, score in results) / len(results)
            
            return QueryResult(
                response = response,
                sources = sources,
                confidence = avg_confidence,
                num_sources = len(sources)
            )
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return QueryResult(response=f"Error processing query: {e}")
    
    def evaluate_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """ Evaluate similarity between two texts """
        evaluator = CosineEmbeddingEvaluator(self.vector_store_manager.embedding_function)
        return evaluator.evaluate_string_pairs(text1, text2)