from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredFileLoader, 
)

import json
import hashlib
from pathlib import Path
from typing import List, Union, Iterator, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import RAGConfig, DEFAULT_RAG_CONFIG
from src.embedding import Embeddings
from src.document.chunking import get_chunker

import logging
logger = logging.getLogger(__name__)

def load_document_worker(file_path: str) -> List[Document]:
    """
    Loads a single document based on its file extension.
    NOTE: This function is designed to be run in a separate process and 
    defined at the top level so it can be pickled by multiprocessing.
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path, strategy="fast")
        
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["file_type"] = file_ext[1:]
        return docs
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []

class DocumentProcessor: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG) -> None:
        self.config = config
        self.cache_path = Path(self.config.corpus_path) / '.cache.json'
    
    def _get_file_hash(self, file_path: Path) -> str:
        """OPTIMIZATION: Get the hash of file content by reading in chunks."""
        h = hashlib.md5()
        with file_path.open("rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def load_cache(self) -> Dict[str, str]:
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: Dict[str, str]) -> None:
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    def load_documents(self, force_reload: bool = False) -> List[Document]: 
        """ Load documents from the corpus """
        documents: List[Document] = []
        cache = self.load_cache() if not force_reload else {}
        new_cache = {}
        
        corpus_path = Path(self.config.corpus_path)
        if not corpus_path.exists(): 
            raise FileNotFoundError(f"Corpus path {self.config.corpus_path} not found.")
        
        # Filter unsupported files 
        files_to_process = [
            f for f in corpus_path.rglob("*") 
            if f.is_file() and f.suffix.lower() in self.config.DocProcessor.supported_extensions
        ]
        
        # Faster loading with multiprocessing 
        with ProcessPoolExecutor(max_workers=8) as executor: 
            future_to_file = {}
            for file_path in files_to_process: 
                file_id = str(file_path)
                try: 
                    current_hash = self._get_file_hash(file_path)
                    new_cache[file_id] = current_hash

                    if not force_reload and cache.get(file_id) == current_hash:
                        logger.info(f"Skipping cached and unchanged file: {file_path.name}")
                        continue

                    # Submit the top-level worker function
                    future = executor.submit(load_document_worker, str(file_path))
                    future_to_file[future] = file_path
                except Exception as e:
                    logger.error(f"Failed to submit {file_path.name} for processing: {e}")

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    docs = future.result()
                    documents.extend(docs)
                    logger.info(f"Successfully loaded {len(docs)} document pages from {file_path.name}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing {file_path.name}: {e}", exc_info=True)
                
        self._save_cache(new_cache)
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]: 
        """
        Split the documents into chunks.
        The chunker is created here, just-in-time, to avoid pickling issues.
        """
        try:    
            # Embedding model is only needed for chunking, so we don't initialize it as a class param
            # Also doing this to keep the DocumentProcessor instance pickleable.
            embeddings = Embeddings(self.config)
            chunker = get_chunker(self.config, embeddings)
            
            chunks = chunker.split_documents(documents=documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
            return chunks
        except Exception as e: 
            logger.error(f"Error splitting documents: {e}", exc_info=True)
            raise