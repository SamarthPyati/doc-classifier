import logging
import yaml
import sys

from dataclasses import dataclass
from enum import Enum
from typing import Literal

# Logger Config (file and stdout logger)
file_handler = logging.FileHandler(filename='events.log')
file_handler.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    datefmt="%d-%m-%Y %H:%M:%S", 
    handlers=[file_handler, stdout_handler]
)
logger = logging.getLogger(__name__)


class LLMModel(Enum):
    GEMMA = "gemma3:4b"
    LLAMA = "llama3.2:3b"

@dataclass
class RAGConfig: 
    """ Configuration for RAG """
    corpus_path: str = "corpus"
    llm_model: str = LLMModel.LLAMA.value
    
    class DocProcessor: 
        # Chunk size to split the document in (Makes the model get precise context to answer better)
        chunk_size: int = 2000
        
        # Overlap token with adjancent chunks
        chunk_overlap: int = 250

        # PDF specific settings
        # pdf_strategy: str = "hi_res"  # "fast", "hi_res", "ocr_only" (for UnstructuredPDFLoader)
        pdf_extract_images: bool = True
        pdf_table_structure_infer_mode: Literal['csv', 'markdown', 'html', None] = 'csv'

        # Supported file extensions
        supported_extensions: tuple = (".pdf", ".txt", ".md", ".xhtml", ".html", ".docx")

    class Database: 
        # Chroma db path to store locally
        database_path: str = "chroma"
        
        # Embedding models (Should be belonging to sentence_transformers) 
        # ["all-MiniLM-L6-v2" (364), "LaBSE" (768), "all-roberta-large-v1" (1024)]
        embedding_model: str = "all-MiniLM-L6-v2"
        
        # Max results to return after similarity search 
        max_results: int = 10

        # Min threshold to classify as related in similarity search
        # NOTE: If a better embedding model is used, increase this to provide much better results, else 
        #       if a tiny model is used lowering would the threshold to about .2 ~ .3 would be good
        similarity_threshold: float = .2

        collection_name: str = "documents"

    class LLM:
        temperature: float = 0.1
        ctx_window_size: int = 2048
        top_p: float = 0.9

    @classmethod
    def load_config_from_yaml(cls, file_path: str): 
        """ Load configuration externally from a yaml file """
        with open(file_path, 'r') as f: 
            config = yaml.safe_load(f)
        return cls(**config)

DEFAULT_RAG_CONFIG: RAGConfig = RAGConfig()

