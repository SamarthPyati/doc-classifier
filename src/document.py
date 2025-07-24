from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredFileLoader, 
)

import hashlib
from pathlib import Path
from typing import List, Union, Iterator

from .config import RAGConfig, DEFAULT_RAG_CONFIG

import logging
logger = logging.getLogger(__name__)

class DocumentProcessor: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG) -> None:
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.DocProcessor.chunk_size, 
            chunk_overlap=self.config.DocProcessor.chunk_overlap, 
            separators=['\n\n', '\n', '', ' '], 
            add_start_index=True,   # For getting offset of file at split
            length_function=len,
            is_separator_regex=False,
        )
        self.processed_files: set = set()

    def _get_document_hash(self, document: Document) -> str: 
        """ Get the hash of a Document required for checking change in document content """
        return hashlib.md5((document.page_content).encode()).hexdigest()
    
    def _get_file_hash(self, file_path: str, size: int = 4096) -> str: 
        """ Get the hash of first 'size' bytes of file content """
        with open(file_path, 'rb') as f: 
            return hashlib.md5(f.read(size)).hexdigest()
    
    def _load_pdf(self, file_path: str, lazy_load: bool = False) -> Union[List[Document], Iterator[Document]]:  
        """ Load a PDF file """
        try: 
            loader = PyMuPDFLoader(
                    file_path,
                    extract_images=self.config.DocProcessor.pdf_extract_images, 
                    extract_tables=self.config.DocProcessor.pdf_table_structure_infer_mode
                )
            docs = loader.load() if not lazy_load else loader.lazy_load()
            
            # Update file metadata 
            file_hash = self._get_file_hash(file_path)
            for doc in docs: 
                doc.metadata.update({
                    "file_hash": file_hash, 
                    "file_type": "pdf"
                })
            return docs
        except Exception as e: 
            logger.error(f"Error in loading PDF file {file_path}: {e}")
            return []

    def _load_document(self, file_path: str, lazy_load: bool = False) -> Union[List[Document], Iterator[Document]]:  
        """ Load single document with appropriate loader """
        file_ext = Path(file_path).suffix.lower()
        try: 
            match file_ext: 
                case ".pdf":
                    return self._load_pdf(file_path, lazy_load=lazy_load) 
                case _: 
                    loader = UnstructuredFileLoader(file_path=file_path, strategy="fast")
                    docs = loader.load() if not lazy_load else loader.lazy_load()

                    # Add metadata
                    file_hash = self._get_file_hash(file_path)
                    for doc in docs:
                        doc.metadata.update({
                            'file_hash': file_hash,
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
        try:
            corpus_path = Path(self.config.corpus_path)
            if not corpus_path.exists(): 
                raise FileNotFoundError(f"Corpus path {self.config.corpus_path} not found.")
            
            for file_path in corpus_path.rglob("*"): 
                if not file_path.is_file(): 
                    continue
                    
                if file_path.suffix.lower() not in self.config.DocProcessor.supported_extensions: 
                    logger.warning(f"Unsupported file format: {file_path.name}")
                    continue

                # Check if file already processed (unless force reload)
                file_key = f"{file_path}_{self._get_file_hash(str(file_path))}"
                if not force_reload and file_key in self.processed_files:
                    logger.info(f"Skipping already processed file: {file_path.name}")
                    continue

                docs = list(self._load_document(str(file_path)))
                if docs is not None: 
                    documents.extend(docs)
                    processed_count += 1
                    self.processed_files.add(file_key)
                    logger.info(f"Loaded {len(docs)} from file {file_path}")
                    
            logger.info(f"Total files processed: {processed_count}")
            logger.info(f"Total documents loaded: {len(documents)}")
            return documents
        
        except Exception as e: 
            logger.error(f"Error loading documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]: 
        """ Split the documents into chunks which are smaller constituent documents """
        try: 
            chunks = self.text_splitter.split_documents(documents=documents)
            return chunks
        except Exception as e: 
            logger.error(f"Error splitting documents: {e}")
            raise