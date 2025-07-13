from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader

from typing import List, Union, Iterator
import logging

from config import RAGConfig, DEFAULT_RAG_CONFIG

# Logger config 
logging.basicConfig(
    filename="events.log", 
    level=logging.INFO, 
)
logger = logging.getLogger(__name__)

class DocumentProcessor: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG) -> None:
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.DocProcessor.chunk_size, 
            chunk_overlap=config.DocProcessor.chunk_overlap, 
            separators=['\n\n', '\n', '', ' '], 
            add_start_index=True,   # For getting offset of file at split
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self, lazy_load: bool = False) -> Union[List[Document], Iterator[Document]]: 
        """ Load documents from corpus """
        try: 
            loader: DirectoryLoader = DirectoryLoader(
                path=self.config.corpus_path, 
                glob="*.xhtml", 
                use_multithreading=True)
            
            documents = list(loader.load()) if not lazy_load else loader.lazy_load()
            logger.info(f"Loaded {len(documents)} documents from {self.config.corpus_path}")

            return documents

        except Exception as e: 
            logger.error("Error loading documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]: 
        """ Split the documents into chunks which are smaller constituent documents """
        try: 
            result = self.text_splitter.split_documents(documents=documents)
            return result
        except Exception as e: 
            logger.error("Error splitting documents: {e}")
            raise