from langchain.schema import Document
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredEPubLoader,
    UnstructuredXMLLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
 
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
from src.constants import ChunkerType

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
        
        # Use specific loaders for better handling of certain file types
        if file_ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        elif file_ext in [".json"]:
            # JSONLoader requires a jq_schema parameter, use UnstructuredLoader for simplicity
            loader = UnstructuredLoader(file_path, strategy="fast")
        elif file_ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
        elif file_ext == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_ext == ".epub":
            loader = UnstructuredEPubLoader(file_path)
        elif file_ext == ".xml":
            loader = UnstructuredXMLLoader(file_path)
        elif file_ext == ".rtf":
            loader = UnstructuredRTFLoader(file_path)
        elif file_ext == ".odt":
            loader = UnstructuredODTLoader(file_path)
        else:
            # Default to UnstructuredLoader for other supported formats
            # (txt, md, html, xhtml, docx, yaml, yml)
            loader = UnstructuredLoader(file_path, strategy="fast")
        
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
    
    def _get_file_hash_sync(self, file_path: Path) -> str:
        """ Synchronous version of file hash for use in multiprocess mode """
        h = hashlib.md5()
        with file_path.open("rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    
    async def _get_file_hash(self, file_path: Path) -> str:
        """ Get the hash of file content by reading in chunks. Kept async so as to not block the event loop. """
        loop = asyncio.get_running_loop() 
        return await loop.run_in_executor(None, self._get_file_hash_sync, file_path)


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
        # Disable multiprocessing for Semantic Chunking due to gRPC/asyncio issues in worker processes
        if self.config.DocProcessor.chunker_type == ChunkerType.SEMANTIC_CHUNKER:
            if multiprocess:
                logger.info("Disabling multiprocessing for Semantic Chunking to avoid gRPC threading issues.")
            multiprocess = False

        documents: List[Document] = []
        cache = self.load_cache() if not force_reload else {}
        new_cache = {}
        
        corpus_path = Path(self.config.corpus_path)
        if not corpus_path.exists(): 
            raise FileNotFoundError(f"Corpus path {self.config.corpus_path} not found.")
        
        # Filter unsupported files and exclude system/hidden files
        files_to_process = [
            f for f in corpus_path.rglob("*") 
            if f.is_file() 
            and f.suffix.lower() in self.config.DocProcessor.supported_extensions
            and not f.name.startswith(".")  # Exclude hidden files (e.g., .cache.json)
            and f.name != ".cache.json"  # Explicitly exclude cache file
        ]
        
        # MULTIPROCESS
        if multiprocess:
            worker_count: int = os.cpu_count() or 4
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_file = {}
                for file_path in files_to_process:
                    file_id = str(file_path)
                    try:
                        # Use sync version in multiprocess mode
                        current_hash = self._get_file_hash_sync(file_path)
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
                        # Continue processing even if a file returns empty docs
                        if docs:
                            documents.extend(docs)
                            logger.info(f"Successfully loaded {len(docs)} document pages from {file_path.name}")
                        else:
                            logger.warning(f"No content loaded from {file_path.name}")
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

            # This line removes metadata that is not a string, integer, float, or boolean.
            # This is necessary because ChromaDB does not support complex metadata types.
            filtered_chunks = filter_complex_metadata(chunks)

            logger.info(f"Split {len(documents)} documents into {len(filtered_chunks)} chunks.")
            return filtered_chunks
        except Exception as e: 
            logger.error(f"Error splitting documents: {e}", exc_info=True)
            raise