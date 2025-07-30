from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, BasePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from .document import DocumentProcessor
from .database import VectorStoreManager
from .config import RAGConfig, DEFAULT_RAG_CONFIG, LLMModel
from .models import RAGContext, Result
from .session import SessionStore

import time
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import functools

import logging
logger = logging.getLogger(__name__)

PROMPTS: Dict[str, BasePromptTemplate] = {
    "query": PromptTemplate( 
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
    ), 

    "chat": ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI assistant engaged in a conversation. Use the provided context and conversation history to give accurate, helpful responses.

            Context Information:
            {context}

            Instructions:
            - Use the context and conversation history to provide relevant answers
            - Be conversational and natural while staying accurate
            - Reference previous parts of the conversation when relevant
            - If the question relates to earlier topics, acknowledge that connection
            - If context is insufficient for a complete answer, ask clarifying questions
            - Maintain consistency with previous responses
            - Always cite sources when providing specific information"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
    ])
}
    
class RAGSystem: 
    """ Main RAG System orchestrator """
    # TODO: Implement chat feature 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):     
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)

        # Make a uuid for current chat session
        self.current_session_id = str(uuid.uuid4())
        self.session_store = SessionStore()

        self._db = None
        self._llm = None

        # Build LCEL chains
        self.query_chain = self._build_query_chain()
        self.conversational_chain = self._build_conversational_chain()

    # Lazy Getters for slow-to-initialize objects
    def _get_llm(self):
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm

    def _get_db(self):
        if self._db is None:
            self._db = self.vector_store_manager.load_vector_store()
        return self._db

    def _initialize_llm(self) -> ChatGoogleGenerativeAI | OllamaLLM | None:
        """ Initialize the LLM based on configuration """
        try:
            model_name = self.config.LLM.model.value
            
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

    # Cache repeated queries
    @functools.lru_cache(maxsize=128)                       
    def _retrieve_context(self, query: str) -> RAGContext:
        """ Retrieve relevant documents using vector search """
        try:
            results = self.vector_store_manager.similarity_search(query, self._get_db())
            
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

    def _build_query_chain(self):
        """Build LCEL chain for single queries without conversation history"""

        query_prompt = PROMPTS.get("query")
        
        chain = (
            RunnablePassthrough()
            | query_prompt 
            | self._get_llm()
            | StrOutputParser()
        )
        
        return chain

    # TODO: Automatically detect new document upload and rebuild the knowledge base (try with watchdog and make a filemonitor)
    def build_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """ Build the knowledge base i.e index the database from documents """
        try: 
            start_time = time.perf_counter()
            logger.info("Building knowledge base...")
            
            # Load and process documents
            documents = self.document_processor.load_documents(force_reload=force_rebuild)
            if not documents:
                logger.info("No new documents found to process")
                return True
            
            # Split documents into chunks
            chunks = self.document_processor.split_documents(documents)

            success = self.vector_store_manager.add_documents(chunks, force_rebuild=force_rebuild)
            
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
    
    def query(self, question: str) -> Result:
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

    def _build_chat_chain(self):
        """Build LCEL chain for chat with conversation history"""
        
        chat_prompt = PROMPTS.get("chat")
        
        # Build the chain with context retrieval
        def prepare_chat_input(inputs):
            question = inputs["question"]
            history = inputs.get("history", [])
            
            # Retrieve context based on current question
            rag_context = self._retrieve_context(question)
            formatted_context = self._format_context(rag_context)
            
            return {
                "context": formatted_context,
                "question": question,
                "history": history,
                "rag_context": rag_context  # Pass for metadata
            }
        
        chain = (
            RunnableLambda(prepare_chat_input)
            | RunnableParallel({
                "response": chat_prompt | self._get_llm() | StrOutputParser(),
                "rag_context": RunnableLambda(lambda x: x["rag_context"])
            })
        )
        
        return chain

    def _build_conversational_chain(self):
        """ Build conversational chain with message history management """ 
        
        # Wrap the chat chain with message history
        conversational_chain = RunnableWithMessageHistory(
            self._build_chat_chain(),
            self.session_store.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            output_messages_key="response"
        )
        
        return conversational_chain

    # IDEATE: How will this chat system work?
    # Every time user queries the system, it will essentially perform the search operation on the document corpus once again. 
    # Optimal approach would be to chat query the entire corpus once and store that context into the conversation (Make a conversation 
    # class) and use that to answer the follow up questions. 
    def chat(self, message: str, session_id: Optional[str] = None) -> Result:
        """ Process a chat message with conversation history """
        try:
            if session_id is None:
                session_id = self.current_session_id
            
            start_time = time.time()
            
            # Configure the runnable with session
            config = {"configurable": {"session_id": session_id}}
            
            # Run the conversational chain
            result = self.conversational_chain.invoke(
                {"question": message},
                config=config
            )
            
            processing_time = time.time() - start_time
            
            # Save the current session to disk
            self.session_store._save_to_disk()
            
            return Result(
                response = result["response"],
                sources = result["rag_context"].sources,
                confidence = result["rag_context"].confidence,
                num_sources = len(result["rag_context"].sources),
                processing_time = processing_time,
                session_id = session_id
            )
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return Result(response=f"Error processing message: {str(e)}")

    def stream_chat(self, message: str, session_id: Optional[str] = None):
        """ Stream chat response for real-time interaction """
        if session_id is None:
            session_id = self.current_session_id
        
        config = {"configurable": {"session_id": session_id}}
        
        try:
            for chunk in self.conversational_chain.stream(
                {"question": message}, 
                config=config
            ):
                if "response" in chunk:
                    yield chunk["response"]
            
            self.session_store._save_to_disk()
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
        
        self.session_store.clear_session(session_id)
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
        return self.vector_store_manager.get_docs_count()