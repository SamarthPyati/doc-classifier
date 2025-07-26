from enum import Enum

class LLMModel(str, Enum):
    # Ollama models (run locally)
    GEMMA = "gemma3:4b"
    LLAMA = "llama3.2:3b"

    # Google gemini models 
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-2.5-pro"

class LLMModelProvider(str, Enum):
    GOOGLE = "google_genai"

class EmbeddingProvider(str, Enum):
    HUGGINGFACE = "HuggingFace"
    GOOGLE = "Google-Gemini"
    OPENAI = "Openai"

class EmbeddingModelHuggingFace(str, Enum):
    MINI_LM = "all-MiniLM-L6-v2"            # Embedding length - 364
    LABSE   = "LaBSE"                       # Embedding length - 728
    ROBERTA = "all-roberta-large-v1"        # Embedding length - 1024
