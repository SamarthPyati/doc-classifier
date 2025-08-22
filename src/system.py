from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig 

from .database import VectorStoreManager
from .document import DocumentProcessor
from .llm.chains import ChainFactory
from .config import RAGConfig, DEFAULT_RAG_CONFIG
from .llm import LLMFactory
from .models import RAGContext, Result
from .session import SessionStore

import time
import uuid
from typing import List, Dict, Optional, Any
import functools

import logging
logger = logging.getLogger(__name__)
    
class RAGSystem: 
    """ Main RAG System orchestrator """
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):     
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.llm_factory = LLMFactory(config)
        self.vector_store = VectorStoreManager(config).get_vector_store()

        # Make an uuid for current chat session
        self.current_session_id = str(uuid.uuid4())
        self.session_store = SessionStore()
        
        self._llm = None

        # TODO: Lazy initialization of chain_factory and other chains
        chain_factory = ChainFactory(
            llm=self._get_llm(), 
            retrieve_context_f=self._retrieve_context,
            format_context_f=self._format_context
        )

        self.query_chain = chain_factory.create_query_chain()
        self.conversational_chain = chain_factory.create_conversational_chain(self.session_store)

    # Lazy Getters for slow-to-initialize objects
    def _get_llm(self):
        if self._llm is None:
            self._llm = self.llm_factory.get_llm()
        return self._llm

    # Cache repeated queries
    @functools.lru_cache(maxsize=128)                       
    def _retrieve_context(self, query: str) -> RAGContext:
        """ Retrieve relevant documents using vector search """
        try:
            results = self.vector_store.similarity_search(query)
            
            if not results:
                return RAGContext()
            
            documents = [doc for doc, _score in results]
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in documents]))
            confidence = sum(score for _, score in results) / len(results)
            
            return RAGContext(
                documents=documents,
                sources=sources,
                confidence=confidence,
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return RAGContext()

    def _format_context(self, context: RAGContext) -> str:
        """ Format retrieved documents into context string """
        if not context.documents:
            return "No relevant documents found."
        
        formatted_docs = [doc.page_content for doc in context.documents]
        
        return ("\n" + "=" * 100 + "\n").join(formatted_docs)

    async def build_knowledge_base(self, multiprocess: bool = False, force_rebuild: bool = False) -> bool:
        """ Build the knowledge base i.e. index the database from documents """
        try: 
            start_time = time.perf_counter()
            logger.info("Building knowledge base ...")
            
            # Reset database if overwrite flag is set 
            if force_rebuild:
                logger.warning("Overwrite flag is set. Resetting the vector store.")
                reset = self.vector_store.reset()
                if not reset:
                    logger.error("Failed to reset the vector store. Aborting build.")
                    return False

            # Load and process documents
            documents = await self.document_processor.load_documents(multiprocess=multiprocess, force_reload=force_rebuild)
            if not len(documents):
                logger.info("No new documents found to process")
                return True
            
            # Split documents into chunks
            chunks = self.document_processor.split_documents(documents)

            success = self.vector_store.add_documents(chunks, force_rebuild=force_rebuild)
            
            if success:
                build_time = time.perf_counter() - start_time
                logger.info(f"Knowledge base built successfully in {build_time:.2f} seconds")
                logger.info(f"Total chunks indexed: {len(chunks)}")
                logger.info("=" * 80)
                return True
            else:
                logger.error("Failed to create vector store", exc_info=True)
                return False

        except Exception as e: 
            logger.error(f"Error building knowledge Base: {e}", exc_info=True)
            return False
    
    async def query(self, question: str) -> Result:
        """ Query the RAG system with a question """
        try:    
            start: float = time.perf_counter()
            
            context: RAGContext = self._retrieve_context(question)

            try:
                response = await self.query_chain.ainvoke({
                    'question': question, 
                    'context': self._format_context(context)
                })

            except Exception as e:
                logger.error(f"LLM invocation failed: {e}. If using OLLAMA check if the server is started.")
                return Result(
                    response="Sorry, I encountered an error while generating the response. Please try again."
                )

            return Result(
                response = response,
                session_id=self.current_session_id,
                sources = context.sources,
                confidence = context.confidence,
                num_sources = len(context.sources), 
                processing_time = time.perf_counter() - start
            )

        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return Result(response=f"Error processing query: {e}")

    # IDEATE: How will this chat system work?
    # Every time user queries the system, it will essentially perform the search operation on the document corpus once again. 
    # Optimal approach would be to chat query the entire corpus once and store that context into the conversation (Make a conversation 
    # class) and use that to answer the follow-up questions.
    async def chat(self, message: str, session_id: Optional[str] = None) -> Result:
        """ Process a chat message with conversation history """
        session_id = session_id or self.current_session_id
        start_time = time.time()

        try:
            # Configure the runnable with session
            config = RunnableConfig({"configurable": {"session_id": session_id}})
            
            # Run the conversational chain
            result = await self.conversational_chain.ainvoke(
                {"question": message},
                config=config
            )
            
            processing_time = time.time() - start_time
            
            return Result(
                response = result["response"],
                sources = result["rag_context"].sources,
                confidence = result["rag_context"].confidence,
                num_sources = len(result["rag_context"].sources),
                processing_time = processing_time,
                session_id = session_id
            )
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}", exc_info=True)
            return Result(response=f"Error processing message: {str(e)}")

    async def stream_chat(self, message: str, session_id: Optional[str] = None):
        """ Stream chat response for real-time interaction """
        session_id = session_id or self.current_session_id
        
        config = RunnableConfig({"configurable": {"session_id": session_id}})
        
        try:
            async for chunk in self.conversational_chain.astream(
                {"question": message}, 
                config=config
            ):
                if "response" in chunk:
                    yield chunk["response"]
            
        except Exception as e:
            logger.error(f"Error streaming chat: {e}")
            yield f"Error: {str(e)}"

    def get_chat_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """ Get chat history for a session """
        if session_id is None:
            session_id = self.current_session_id
        
        history = self.session_store.get_session_history(session_id)
        
        formatted_history = []
        for message in history.messages:
            formatted_history.append({
                "role": "human" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
                "timestamp": getattr(message, 'timestamp', None)
            })
        
        return formatted_history

    def clear_chat_history(self, session_id: Optional[str] = None):
        """ Clear chat history for a session """
        if session_id is None:
            session_id = self.current_session_id
        
        self.session_store.delete_session(session_id)
        logger.info(f"Cleared chat history for session: {session_id}")

    def create_new_session(self) -> str:
        """ Create a new chat session """
        new_session_id = str(uuid.uuid4())
        self.session_store.new_session(new_session_id)
        logger.info(f"Created new chat session: {new_session_id}")
        return new_session_id

    def list_sessions(self) -> List[str]:
        """ List all active sessions """
        return self.session_store.list_sessions()

    def document_count(self) -> int: 
        return self.vector_store.count()

    def list_document(self, n: int) -> None:
        self.vector_store.peek(n) 

# TODO: Automatically detect new document upload and rebuild the knowledge base (try with watchdog and make a filemonitor)