from langchain.schema import Document
from langchain_chroma import Chroma
from chromadb.config import Settings

from .config import RAGConfig, DEFAULT_RAG_CONFIG
from .embedding import Embeddings

import os
import shutil
from pathlib import Path
from typing import List, Iterator, Tuple
from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)

load_dotenv()

class VectorStoreManager: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config 
        self.database_path = os.path.abspath(self.config.Database.path)
        self.collection_name = self.config.Database.collection_name
        self.embedding_function = Embeddings(config).get_embedding_model()
    
        self._db = self.load_vector_store()

    def load_vector_store(self) -> Chroma:
        """ Get or create ChromaDB client with proper settings """
        try: 
            if not Path(self.database_path).exists():
                # If chroma doesn`t exists create it
                logger.warning(f"Vector store not found at {self.database_path}. Creating a new one ...")

            db = Chroma(
                collection_name=self.collection_name, 
                persist_directory=self.database_path,
                embedding_function=self.embedding_function,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get collection info
            item_count = db._collection.count()
            logger.info(f"Loaded vector store with {item_count} documents from {self.database_path}")
            return db

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Try to reset and return None
            self.reset_database()
            return None

    def reset_database(self) -> bool:
        """ Reset/clear the entire database """
        try:
            if Path(self.database_path).exists():
                shutil.rmtree(self.database_path)
                logger.info("Database reset successfully")
            self._db = None
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

    def calculate_chunk_ids(self, chunks: List[Document]):
        """ Assigns unique chunk IDs to each Document in the provided list based on their source and page metadata """
        # This will create IDs like "procurement.pdf:6:2"
        # ID Structure Format => "Page Source:Page Number:Chunk Index"
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

    def __batch_list(self, documents: List[Document], batch_size: int) -> Iterator[Tuple[List[Document], List[str]]]:
        """ Yield successive n-sized chunks and ids tuple from a list. Helper function for chroma db to keep batch size less than 5461. """
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = [doc.metadata["id"] for doc in batch]
            yield batch, ids


    def add_documents(self, chunks: List[Document], force_rebuild: bool = False) -> bool:
        """ Add new documents to existing vector store """
        try: 
            if not self._db:
                logger.error("No vector store available to add documents")
                return False

            # Get chunks with ids 
            self.calculate_chunk_ids(chunks)

            new_chunks: List[Document] = []    
            new_ids: List[str] = []

            if not force_rebuild: 
                # Filter out existing documents to avoid rebuilding 
                existing_items = self._db.get(include=[])
                existing_ids = set(existing_items["ids"])
                logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

                for chunk in chunks:    
                    if chunk.metadata["id"] not in existing_ids: 
                        new_ids.append(chunk.metadata["id"])
                        new_chunks.append(chunk)
            else: 
                # If force_rebuild enabled just build everything from scratch
                new_chunks = chunks
                new_ids = [chunk.metadata["id"] for chunk in chunks]
            
            if len(new_chunks):
                logger.info(f"Adding {len(new_chunks)} new chunks to the database")
                for batch, ids in self.__batch_list(new_chunks, batch_size=1000): 
                    self._db.add_documents(batch, ids=ids)
                return True
            else:
                logger.info("No new documents to add")
                return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

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