from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
import chromadb

from config import RAGConfig, DEFAULT_RAG_CONFIG, logger

import os
import shutil
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config 
        self.embedding_function = None
        self._db = None
        self._client = None

        # Initialize embedding function
        try:
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=self.config.Database.embedding_model, 
                model_kwargs={'device': 'cpu'},  
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embedding model: {self.config.Database.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

        self.database_path = os.path.abspath(self.config.Database.database_path)
        self.collection_name = self.config.Database.collection_name

    def _get_chroma_client(self):
        """ Get or create ChromaDB client with proper settings """
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.database_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._client

    def reset_database(self) -> bool:
        """ Reset/clear the entire database """
        try:
            if Path(self.database_path).exists():
                shutil.rmtree(self.database_path)
                logger.info("Database reset successfully")
            self._db = None
            self._client = None
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

    def create_vector_store(self, documents: List[Document], ids: Optional[List[str]] = None, overwrite: bool = False) -> bool:
        """ Create a vector database from the documents """
        try: 
            if overwrite and Path(self.database_path).exists():
                self.reset_database()

            if not documents: 
                logger.warning("No documents to index")
                return False

            # Calculate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]

            # Create new vector store
            self._db = Chroma.from_documents(
                documents=documents, 
                embedding=self.embedding_function,
                ids=ids, 
                collection_name=self.collection_name, 
                persist_directory=self.database_path,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            logger.info(f"Created vector store with {len(documents)} documents at {self.database_path}")
            return True

        except Exception as e: 
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def calculate_chunk_ids(self, chunks: List[Document]) -> List[str]: 
        """ Calculate unique IDs for document chunks """
        ids = []
        last_page_id = None 
        current_chunk_index = 0

        for i, chunk in enumerate(chunks): 
            source = chunk.metadata.get('source', 'Unknown')
            page = chunk.metadata.get('page', '0')
            current_page_id = f"{source}:{page}"
            
            if current_page_id == last_page_id: 
                current_chunk_index += 1
            else: 
                current_chunk_index = 0 
            
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["chunk_id"] = chunk_id
            ids.append(chunk_id)

        return ids

    def add_documents(self, chunks: List[Document]) -> bool:
        """ Add new documents to existing vector store """
        try: 
            if not self._db:
                self.load_vector_store()
            
            if not self._db:
                logger.error("No vector store available to add documents")
                return False

            # Calculate chunk IDs
            ids = self.calculate_chunk_ids(chunks)
            
            # Get existing IDs to avoid duplicates
            try:
                existing_items = self._db.get(include=[])
                existing_ids = set(existing_items["ids"])
                logger.info(f"Number of existing documents in DB: {len(existing_ids)}")
            except:
                existing_ids = set()

            # Filter out existing documents
            new_chunks = []
            new_ids = []
            for chunk, chunk_id in zip(chunks, ids):
                if chunk_id not in existing_ids:
                    new_chunks.append(chunk)
                    new_ids.append(chunk_id)

            if new_chunks:
                logger.info(f"Adding {len(new_chunks)} new documents")
                self._db.add_documents(new_chunks, ids=new_ids)
                return True
            else:
                logger.info("No new documents to add")
                return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def load_vector_store(self) -> Optional[Chroma]:
        """ Load existing vector store """
        try:
            if not Path(self.database_path).exists():
                logger.warning(f"Vector store not found at {self.database_path}")
                return None
            
            self._db = Chroma(
                collection_name=self.collection_name, 
                persist_directory=self.database_path,
                embedding_function=self.embedding_function,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get collection info
            item_count = self._db._collection.count()
            logger.info(f"Loaded vector store with {item_count} documents from {self.database_path}")
            
            return self._db
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Try to reset and return None
            self.reset_database()
            return None

    def similarity_search(self, query: str, db: Chroma) -> List[tuple[Document, float]]: 
        """ Perform similarity search with a query """
        try:
            if not db:
                logger.error("Database was not properly initialized")
                return []

            # Perform similarity search
            results = db.similarity_search_with_relevance_scores(
                query, 
                k=self.config.Database.max_results
            )

            # Filter by threshold and sort by relevance
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= self.config.Database.similarity_threshold
            ]
            
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(filtered_results)} relevant documents for query: \"{query[:50]}...\"")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []