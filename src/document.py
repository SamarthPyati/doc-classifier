from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredFileLoader, 
)

import json
import hashlib
from pathlib import Path
from typing import List, Union, Iterator, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import RAGConfig, DEFAULT_RAG_CONFIG
from .embedding import Embeddings

import logging
logger = logging.getLogger(__name__)

class DocumentProcessor: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG) -> None:
        self.config = config
        self.embeddings = Embeddings(config)
        self.cache_path = Path(self.config.corpus_path) / '.cache.json'

        if self.config.DocProcessor.use_semantic_chunking:
            self.text_splitter = SemanticChunker(
                embeddings=self.embeddings.get_embedding_model(),
                add_start_index=True, 
                breakpoint_threshold_type="standard_deviation", 
                min_chunk_size=self.config.DocProcessor.chunk_size, 
            )
        else: 
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.DocProcessor.chunk_size, 
                chunk_overlap=self.config.DocProcessor.chunk_overlap, 
                separators=['\n\n', '\n', '.', '?', '!', ' ', ''], 
                add_start_index=True,   # For getting offset of file at split
                length_function=len,
                is_separator_regex=False,
            )
    
    def _get_file_hash(self, file_path: Path) -> str:
        """ OPTIMIZATION: Get the hash of file content by reading in chunks """
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
    
    def _load_pdf(self, file_path: str, lazy_load: bool = False) -> Union[List[Document], Iterator[Document]]:  
        """ Load a PDF file """
        try: 
            loader = PyMuPDFLoader(
                    file_path,
                    extract_images=self.config.DocProcessor.pdf_extract_images, 
                    extract_tables=self.config.DocProcessor.pdf_table_structure_infer_mode
                )
            docs = loader.load() if not lazy_load else loader.lazy_load()
            
            # Add filetype in metadata 
            for doc in docs: 
                doc.metadata.update({
                    "file_type": "pdf"
                })
            return docs
        except Exception as e: 
            logger.error(f"Error in loading PDF file {file_path}: {e}")
            return []

    def _load_single_document(self, file_path: str, lazy_load: bool = False) -> Union[List[Document], Iterator[Document]]:  
        """ Load single document with appropriate loader """
        try: 
            file_ext = Path(file_path).suffix.lower()
            match file_ext: 
                case ".pdf":
                    return self._load_pdf(file_path, lazy_load=lazy_load) 
                case _: 
                    loader = UnstructuredFileLoader(file_path=file_path, strategy="fast")
                    docs = loader.load() if not lazy_load else loader.lazy_load()

                    # Add filetype in metadata
                    for doc in docs:    
                        doc.metadata.update({
                            'file_type': file_ext[1:],  # Remove the dot from extension
                        })
                    return docs

        except Exception as e:
            logger.error(f"Error in loading file {file_path}:{e}")
            return []

    def load_documents(self, force_reload: bool = False) -> List[Document]: 
        """ Load documents from corpus """
        documents: List[Document] = []
        processed_count = 0
        
        cache = self.load_cache() if not force_reload else {}
        new_cache = {}
        
        corpus_path = Path(self.config.corpus_path)
        if not corpus_path.exists(): 
            raise FileNotFoundError(f"Corpus path {self.config.corpus_path} not found.")
        
        files_to_process = list(corpus_path.rglob("*"))
        
        with ProcessPoolExecutor(max_workers=8) as executor: 
            future_to_file = {}             # Future to File map to extract result 
            for file_path in files_to_process: 
                if not file_path.is_file(): 
                    continue
                
                if file_path.suffix.lower() not in self.config.DocProcessor.supported_extensions: 
                    logger.warning(f"Unsupported file format: {file_path.name}")
                    continue

                file_id = str(file_path)
                try: 
                    current_hash = self._get_file_hash(file_path)
                    new_cache[file_id] = current_hash

                    if not force_reload and cache.get(file_id) == current_hash:
                        logger.info(f"Skipping cached and unchanged file: {file_path.name}")
                        continue

                    future = executor.submit(self._load_single_document, file_path)
                    future_to_file[future] = file_path
                except Exception as e:
                    logger.error(f"Failed to submit {file_path.name} for processing: {e}")

            for future in as_completed(list(future_to_file.keys())):
                file_path = future_to_file.get(future, 'Unknown')
                try:
                    docs = future.result()
                    documents.extend(list(docs))
                    logger.info(f"Successfully loaded {len(docs)} document pages from {file_path.name}")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing {file_path.name}: {e}")
                
        self._save_cache(new_cache)
        logger.info(f"Total files processed: {processed_count}")
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]: 
        """ Split the documents into chunks which are smaller constituent documents """
        try: 
            chunks = self.text_splitter.split_documents(documents=documents)
            return chunks
        except Exception as e: 
            logger.error(f"Error splitting documents: {e}")
            raise