from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from document import DocumentProcessor
from database import VectorStoreManager

from config import RAGConfig, DEFAULT_RAG_CONFIG, LLMModel

import time
from typing import List
from dataclasses import dataclass, field

import logging
logger = logging.getLogger(__name__)

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
        self.db = self.vector_store_manager.load_vector_store()
        
        # LLM Initialization
        self.llm = None
        self._initialize_llm()

        self.conversation = []

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
        """Initialize the LLM based on configuration."""
        try:
            model_name = self.config.LLM.llm_model.value

            if model_name.strip().startswith("gemini"):
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name, 
                    temperature=self.config.LLM.temperature,
                    top_p=self.config.LLM.top_p,
                )

            elif model_name.startswith("llama") or model_name.startswith("gemma"):
                try: 
                    self.llm = OllamaLLM(
                        model=model_name,
                        num_thread=self.config.LLM.num_threads,
                        temperature=self.config.LLM.temperature,
                        num_ctx=self.config.LLM.ctx_window_size,
                        top_p=self.config.LLM.top_p,
                        verbose=False
                    )
                except Exception as e:
                    logger.error(f"Ensure that Ollama is running and the model is pulled. Error: {e}")
                    return False
            else:
                logger.warning(f"Invalid LLM model '{model_name}'. Choose from: {', '.join(LLMModel._member_names_)}")
                return False

            # Test the LLM connection
            test_response = self.llm.invoke("Hello")
            if not test_response:
                raise ValueError("Empty response from LLM on test prompt.")

            logger.info(f"LLM initialized successfully with model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return False


    # TODO: Automatically detect new document upload and rebuild the knowledge base
    def build_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """ Build the knowledge base i.e index the database from documents """
        try: 
            start_time = time.perf_counter()
            logger.info("Building knowledge base...")
            
            # Load and process documents
            documents = self.document_processor.load_documents(force_reload=force_rebuild)
            if not documents:
                logger.error("No documents found to process")
                return False
            
            # Split documents into chunks
            chunks = self.document_processor.split_documents(documents)

            success = self.vector_store_manager.add_documents(chunks)
            
            if success:
                build_time = time.perf_counter() - start_time
                logger.info(f"Knowledge base built successfully in {build_time:.2f} seconds")
                logger.info(f"Total chunks indexed: {len(chunks)}")
                logger.info("=" * 80)
                return True
            else:
                logger.error("Failed to create vector store")
                return False

        except Exception as e: 
            logger.error(f"Error building knowledge Base: {e}")
            return False
    
    # IDEATE: How will this query system work?
    # Every time user queries the system, it will essentially perform the search operation on the document corpus once again. 
    # Optimal approach would be to query the entire corpus once and store that context into the conversation (Make a conversation 
    # class) and use that to answer the follow up questions. 
    def query(self, question: str) -> QueryResult:
        """ Query the RAG system with a question """
        try:    
            if not self.llm:
                if not self._initialize_llm():
                    return QueryResult(response="LLM not available. Please check LLM Service.")
            
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
                    if self.config.LLM.llm_model in [LLMModel.GEMINI_FLASH or LLMModel.GEMINI_PRO]: 
                        # Gemini LLM provide response as a big dictionary, so extract content out of it
                        response = self.llm.invoke(prompt).content
                    else: 
                        response = self.llm.invoke(prompt)
                else:
                    response = self.llm.predict(prompt).content

            except Exception as e:
                logger.error(f"LLM invocation failed: {e}. If using OLLAMA check if the server is started.")
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