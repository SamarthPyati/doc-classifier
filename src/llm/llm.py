from src.constants import LLMModelProvider, LLMModel
from src.config import RAGConfig, DEFAULT_RAG_CONFIG
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

import logging
logger = logging.getLogger(__name__)

class LLMFactory: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config
        self.provider = self.config.LLM.provider
        self.model = self.config.LLM.model.value

    def get_llm(self) -> ChatGoogleGenerativeAI | OllamaLLM | None:
        """ Initialize the LLM based on configuration """
        try:
            match self.provider: 
                case LLMModelProvider.GOOGLE: 
                    llm = ChatGoogleGenerativeAI(
                        model=self.model, 
                        temperature=self.config.LLM.temperature,
                        top_p=self.config.LLM.top_p,
                    )
                case LLMModelProvider.OLLAMA:  
                    try: 
                        llm = OllamaLLM(
                            model=self.model,
                            num_thread=self.config.LLM.num_threads,
                            temperature=self.config.LLM.temperature,
                            num_ctx=self.config.LLM.ctx_window_size,
                            top_p=self.config.LLM.top_p,
                            verbose=False
                        )
                    except Exception as e:
                        logger.error(f"Ensure that Ollama is running and the model is pulled. Error: {e}")
                        return None
                case _:     
                    logger.warning(f"Invalid LLM model '{self.model}'. Choose from: {', '.join(LLMModel._member_names_)}")
                    return None

            # Test the LLM connection
            test_response = llm.invoke("Hello")
            if not test_response:
                raise ValueError("Empty response from LLM on test prompt.")

            logger.info(f"LLM initialized successfully with model: {self.model}")
            return llm 

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return None