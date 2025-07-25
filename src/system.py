from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import Document

from .document import DocumentProcessor
from .database import VectorStoreManager
from .config import RAGConfig, DEFAULT_RAG_CONFIG, LLMModel

import time
import uuid
from typing import List, Dict
from dataclasses import dataclass, field

import logging
logger = logging.getLogger(__name__)

@dataclass 
class QueryResult:
    response: str       = "Null"
    sources: List[str]  = field(default_factory=list)
    confidence: float   = 0.0
    num_sources: int    = 0
    generation_time: float = 0.0
    session_id: str = "Null"

    def __repr__(self):
        return (f"Response: {self.response}\n"
                f"Session ID: {self.session_id}\n"
                f"Sources: {self.sources}\n"
                f"Confidence: {self.confidence:.3f}\n"
                f"Number of sources: {self.num_sources}\n"
                f"Generation time: {self.generation_time:.3f} second(s)\n")

@dataclass 
class RAGContext: 
    """Data class to hold RAG context and sources"""
    documents: List[Document] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    query: str = ""


# TODO: Also autogenerate chat history and store in session store 
# TODO: Store the session store in database instead of in memory
class SessionStore: 
    """In-memory session store for chat histories"""
    def __init__(self): 
        self.store: Dict[str, BaseChatMessageHistory] = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory: 
        if not session_id in self.store: 
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def clear_session(self, session_id: str): 
        if session_id in self.store: 
            self.store[session_id].clear()
    
    def list_session(self) -> List[str]: 
        return list(self.store.keys())
    
class RAGSystem: 
    """ Main RAG System orchestrator """
    # TODO: Implement chat feature 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):     
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)

        # Make a uuid for current chat session
        self.current_session_id = str(uuid.uuid4())

        # Load vector store and initialize llm 
        self.db = self.vector_store_manager.load_vector_store()
        self.llm = self._initialize_llm()
        
        # Build LCEL chains
        self.query_chain = self._build_query_chain()

    def _initialize_llm(self) -> ChatGoogleGenerativeAI | OllamaLLM | None:
        """ Initialize the LLM based on configuration """
        try:
            model_name = self.config.LLM.llm_model.value
            
            if model_name.strip().startswith("gemini"):
                llm = ChatGoogleGenerativeAI(
                    model=model_name, 
                    temperature=self.config.LLM.temperature,
                    top_p=self.config.LLM.top_p,
                )
            elif model_name.startswith("llama") or model_name.startswith("gemma"):
                try: 
                    llm = OllamaLLM(
                        model=model_name,
                        num_thread=self.config.LLM.num_threads,
                        temperature=self.config.LLM.temperature,
                        num_ctx=self.config.LLM.ctx_window_size,
                        top_p=self.config.LLM.top_p,
                        verbose=False
                    )
                except Exception as e:
                    logger.error(f"Ensure that Ollama is running and the model is pulled. Error: {e}")
                    return None
            else:
                logger.warning(f"Invalid LLM model '{model_name}'. Choose from: {', '.join(LLMModel._member_names_)}")
                return None

            # Test the LLM connection
            test_response = llm.invoke("Hello")
            if not test_response:
                raise ValueError("Empty response from LLM on test prompt.")
                return None

            logger.info(f"LLM initialized successfully with model: {model_name}")
            return llm 

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return None

    def _retrieve_context(self, query: str) -> RAGContext:
        """ Retrieve relevant documents using vector search """
        try:
            if not self.db:
                self.db = self.vector_store_manager.load_vector_store()
            
            results = self.vector_store_manager.similarity_search(query, self.db)
            
            if not results:
                return RAGContext(query=query)
            
            documents = [doc for doc, _score in results]
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in documents]))
            confidence = sum(score for _, score in results) / len(results)
            
            return RAGContext(
                documents=documents,
                sources=sources,
                confidence=confidence,
                query=query
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return RAGContext(query=query)

    def _format_context(self, context: RAGContext) -> str:
        """ Format retrieved documents into context string """
        if not context.documents:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(context.documents, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted_docs.append(f"Document {i} -> (Source: {source}):\n{doc.page_content}")
        
        return ("\n\n" + "=" * 70 + "\n\n").join(formatted_docs)

    def _build_query_chain(self):
        """Build LCEL chain for single queries without conversation history"""

        query_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert AI assistant that provides accurate, comprehensive and concise answers based on the given context.
        Context Information:
        {context}
        
        Question: {question}
        
        Instructions:
        - Answer using ONLY the provided context
        - Be comprehensive yet concise
        - If context is insufficient, state this clearly, do not make up any answers by yourself
        - Cite specific sources when mentioning details
        - Provide structured, actionable information when possible
        
        Answer:"""
        )
        
        chain = (
            RunnablePassthrough()
            | query_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        return chain

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
            start: float = time.perf_counter()
            
            context: RAGContext = self._retrieve_context(question)

            try:
                response = self.query_chain.invoke({
                    'question': question, 
                    'context': self._format_context(context)
                })

            except Exception as e:
                logger.error(f"LLM invocation failed: {e}. If using OLLAMA check if the server is started.")
                return QueryResult(
                    response="Sorry, I encountered an error while generating the response. Please try again."
                )

            return QueryResult(
                response = response,
                session_id=self.current_session_id,
                sources = context.sources,
                confidence = context.confidence,
                num_sources = len(context.sources), 
                generation_time= time.perf_counter() - start
            )

        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return QueryResult(response=f"Error processing query: {e}")