from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredFileLoader, 
)
 
import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import RAGConfig, DEFAULT_RAG_CONFIG
from src.embedding import Embeddings
from src.document.chunking import get_chunker
from src.document.classification import get_classifier

import logging
logger = logging.getLogger(__name__)

async def load_document_worker(file_path: str, config: RAGConfig = DEFAULT_RAG_CONFIG) -> List[Document]:
    """
    Loads a single document, classifies it, and adds the category to its metadata.
    This function is designed to be run in a separate process.

    NOTE: This function is designed to be run in a separate process and 
    defined at the top level so it can be pickled by multiprocessing.
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path, strategy="fast")
        
        docs = await loader.aload()
        if not docs:
            logger.warning(f"No content loaded from file: {file_path}")
            return []

        # Classification of file 
        category = "Unclassified"
        if config.DocProcessor.enable_classification: 
            classifier = get_classifier(config)
            # Take a sample text for classification
            content = '\n'.join([doc.page_content for doc in docs[:20]])[:8000]
            category = await classifier.classify(content_sample=content)
            # logger.info(f"Classified '{Path(file_path).name}' as: {category}")

        # Add file type in metadata
        for doc in docs:
            doc.metadata["file_type"] = file_ext[1:]
            doc.metadata["file_category"] = category
        return docs
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
        return []

class DocumentProcessor: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG) -> None:
        self.config = config
        self.cache_path = Path(self.config.corpus_path) / '.cache.json'
    
    async def _get_file_hash(self, file_path: Path) -> str:
        """ Get the hash of file content by reading in chunks. Kept async so as to not block the event loop. """
        loop = asyncio.get_running_loop() 
        
        def _hash_sync(file_path: Path) -> str: 
            h = hashlib.md5()
            with file_path.open("rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()
        
        return await loop.run_in_executor(None, _hash_sync, file_path)


    def load_cache(self) -> Dict[str, str]:
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: Dict[str, str]) -> None:
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    async def load_documents(self, multiprocess: bool = False, force_reload: bool = False) -> List[Document]:
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
        
        # TODO: Test whether earlier Mulitprocessing method is faster or new async method.
        # MULTIPROCESS
        if multiprocess:
            worker_count: int | None = os.cpu_count() if not None else 4
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
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
                        if not docs:
                            executor.shutdown()
                        documents.extend(docs)
                        logger.info(f"Successfully loaded {len(docs)} document pages from {file_path.name}")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while processing {file_path.name}: {e}", exc_info=True)
        else:
            # ASYNC
            try:
                tasks = []
                for file_path in files_to_process:
                    file_id = str(file_path)
                    current_hash = await self._get_file_hash(file_path)
                    new_cache[file_id] = current_hash

                    if not force_reload and cache.get(file_id) == current_hash:
                        logger.info(f"Skipping cached and unchanged file: {file_path.name}")
                        continue

                    # Create a task for each document to be loaded and classified
                    tasks.append(load_document_worker(str(file_path), self.config))

                # Run all loading and classification tasks concurrently
                results = await asyncio.gather(*tasks)

                # Flatten the list of lists into a single list of documents
                for doc_list in results:
                    documents.extend(doc_list)
            except Exception as e:
                logger.error(f"Error in load_document_worker(): {e}", exc_info=True)
                return []

        # Update cache
        self._save_cache(new_cache)
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split the documents into chunks.
        The chunker is created here just-in-time to avoid pickling issues.
        """
        try:    
            embeddings = Embeddings(self.config)
            chunker = get_chunker(self.config, embeddings)
            
            chunks = chunker.split_documents(documents=documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
            return chunks
        except Exception as e: 
            logger.error(f"Error splitting documents: {e}", exc_info=True)
            raise