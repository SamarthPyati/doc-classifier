from src.constants import LLMModelProvider
from src.config import RAGConfig, DEFAULT_RAG_CONFIG

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

import logging
logger = logging.getLogger(__name__)

class LLMFactory: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config

    def get_llm(self) -> BaseChatModel | None:
        """ Initialize the LLM based on configuration """
        
        provider = self.config.LLM.provider
        model = self.config.LLM.model.value
        logger.info(f"Initializing LLM with provider: {provider.value} and model: {model}")

        try:
            # Registry based model 
            provider_registry = {
                LLMModelProvider.GOOGLE: self._create_google_llm, 
                LLMModelProvider.OLLAMA: self._create_ollama_llm
            }   

            llm_creator_f = provider_registry.get(provider)
            if not llm_creator_f: 
                logger.error(f"Unsupported LLM provider configured: {provider.value}")
                return None
            
            llm = llm_creator_f()

            # Test the LLM connection
            llm.invoke("Hello")

            logger.info(f"LLM initialized successfully with model: {model}")
            return llm  
        except Exception as e:
            error_message = (
                f"Failed to initialize LLM '{model}' with provider '{provider}'. "
                f"Please check your configuration, API keys, and ensure the model service is running. Original error: {e}"
            )
            logger.error(error_message, exc_info=False)
            return None

    def _create_ollama_llm(self) -> OllamaLLM: 
        return OllamaLLM(
            model=self.config.LLM.model.value, 
            num_thread=self.config.LLM.num_threads,
            temperature=self.config.LLM.temperature,
            num_ctx=self.config.LLM.ctx_window_size,
            top_p=self.config.LLM.top_p,
            verbose=False
        )
    
    def _create_google_llm(self) -> ChatGoogleGenerativeAI: 
        return ChatGoogleGenerativeAI(
            model=self.config.LLM.model.value,  
            temperature=self.config.LLM.temperature,
            top_p=self.config.LLM.top_p,
        )
