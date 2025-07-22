from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from document import DocumentProcessor, CosineEmbeddingEvaluator
from database import VectorStoreManager

from config import RAGConfig, DEFAULT_RAG_CONFIG, logger

import time
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
    # TODO: Implement chat feature 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):     
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)

        # Make prompt template
        self.prompt_template = self._get_prompt_template()

        # Load vector store
        self.db = None
        
        # LLM Initialization
        self.llm = None
        self._initialize_llm()

    def _get_prompt_template(self) -> PromptTemplate: 
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert AI assistant that provides accurate, comprehensive answers based on the given context.
        Context Information:
        {context}
        
        Question: {question}
        
        Instructions:
        - Answer using ONLY the provided context
        - Be comprehensive yet concise
        - If context is insufficient, state this clearly
        - Cite specific sources when mentioning details
        - Provide structured, actionable information when possible
        
        Answer:"""
        )

    def _initialize_llm(self) -> bool: 
        """ Initializing LLM """
        try: 
            self.llm = OllamaLLM(
                model=self.config.llm_model,
                num_thread=self.config.LLM.num_threads,
                temperature=self.config.LLM.temperature, 
                num_ctx=self.config.LLM.ctx_window_size, 
                top_p=self.config.LLM.top_p,
                verbose=False
            )
            
            # Test the LLM connection
            test_response = self.llm.invoke("Hello")
            logger.info(f"LLM initialized successfully with model: {self.config.llm_model}")
            return True
            
        except Exception as e: 
            logger.error(f"Error initializing LLM: {e}")
            logger.error("Make sure Ollama is running and the model is available")
            return False

    # TODO: Automatically detect new document upload and rebuild the knowledge base
    def build_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """ Build the knowledge base from documents """
        try: 
            start_time = time.time()
            logger.info("Building knowledge base...")
            
            # Load and process documents
            documents = self.document_processor.load_documents(force_reload=force_rebuild)
            if not documents:
                logger.error("No documents found to process")
                return False
                
            
            # Split documents into chunks
            chunks = self.document_processor.split_documents(documents)

            chunk_ids = self.vector_store_manager.calculate_chunk_ids(chunks)
            success = self.vector_store_manager.create_vector_store(
                chunks, 
                ids=chunk_ids, 
                overwrite=force_rebuild
            )
            
            if success:
                build_time = time.time() - start_time
                logger.info(f"Knowledge base built successfully in {build_time:.2f} seconds")
                logger.info(f"Total chunks indexed: {len(chunks)}")
                logger.info("=" * 60)
                return True
            else:
                logger.error("Failed to create vector store")
                return False

        except Exception as e: 
            logger.error(f"Error building knowledge Base: {e}")
            return False
        
    def query(self, question: str) -> QueryResult:
        """ Query the RAG system with a question """
        try:    
            if not self.llm:
                if not self._initialize_llm():
                    return QueryResult(response="LLM not available. Please check Ollama service.")
            
            if not self.db: 
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
            
            try:
                if hasattr(self.llm, 'invoke'):
                    response = self.llm.invoke(prompt)
                else:
                    response = self.llm.predict(prompt)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                return QueryResult(
                    response="Sorry, I encountered an error while generating the response. Please try again."
                )
            
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
        return evaluator.evaluate_string_pairs(prediction=text1, prediction_b=text2)