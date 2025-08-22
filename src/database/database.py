from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from chromadb.config import Settings as ChromaSettings

from src.config import RAGConfig, DEFAULT_RAG_CONFIG
from src.constants import VectorStoreProvider
from .interface import VectorStoreInterface

import os
from pathlib import Path
from typing import List, Iterator, Tuple
from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)

load_dotenv()

def calculate_chunk_ids(chunks: List[Document]) -> None:
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

class ChromaManager(VectorStoreInterface): 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.database_path = os.path.abspath(config.Database.path)
        self.collection_name = config.Database.collection_name
        super().__init__(config)

    def _initialize_db(self) -> Chroma:
        """ Get or create ChromaDB client with proper settings """
        try: 
            if not Path(self.database_path).exists():
                # If chroma doesn`t exists create it
                logger.warning(f"Vector store not found at {self.database_path}. Creating a new one ...")

            db = Chroma(
                collection_name=self.collection_name, 
                persist_directory=self.database_path,
                embedding_function=self.embedding_function,
                client_settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True, 
                    is_persistent=True
                )
            )

            # Get collection info
            item_count = db._collection.count()
            logger.info(f"Loaded vector store with {item_count} documents from {self.database_path}")
            return db

        except Exception as e:
            logger.error(f"FATAL: Error loading vector store: {e}", exc_info=True)
            # Raise the exception instead of returning None to prevent silent failures
            raise ConnectionError("Failed to initialize the ChromaDB vector store.") from e

    def _batch_list(self, documents: List[Document], batch_size: int) -> Iterator[Tuple[List[Document], List[str]]]:
        """ 
        Yield successive n-sized chunks and ids tuple from a list. 
        Helper function for chroma db to keep batch size less than 5461 (Chroma DB Limitation).
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = [doc.metadata["id"] for doc in batch]
            yield batch, ids

    async def add_documents(self, chunks: List[Document], force_rebuild: bool = False) -> bool:
        """ Add new documents to existing vector store """
        try: 
            if not self._db:
                self._db = self._initialize_db()

            # Add ids to metadata of chunks
            calculate_chunk_ids(chunks)

            new_chunks: List[Document] = []    

            if not force_rebuild: 
                # Filter out existing documents to avoid rebuilding 
                existing_ids = set(self._db.get(include=[])["ids"])
                new_chunks.extend([chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids])
            else: 
                # If force_rebuild enabled just build everything from scratch
                new_chunks = chunks
            
            if len(new_chunks):
                logger.info(f"Adding {len(new_chunks)} new chunks to the database")
                for batch, ids in self._batch_list(new_chunks, batch_size=1000): 
                    await self._db.aadd_documents(batch, ids=ids)
                return True
            else: 
                return False

        except Exception as e:
            logger.error(f"Error adding documents: {e}", exc_info=True)
            return False

class PineconeManager(VectorStoreInterface):
    def __init__(self, config: RAGConfig):
        self.pinecone_client = Pinecone()
        self.index_name = self.config.Database.collection_name
        super().__init__(config)

    def _initialize_db(self) -> PineconeVectorStore | None:
        try:   
            if not self.pinecone_client.has_index(self.index_name):
                logger.warning(f"Pinecone index '{self.index_name}' not found. Creating...")
                embedding_dim = 768 # TODO: Make a mapping of embedding_model with its dimension

                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws', 
                        region='us-east-1'
                    )
                )

            index = self.pinecone_client.Index(self.index_name)
            return PineconeVectorStore(index, self.embedding_function, "page_content")
        
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Try to reset and return None
            self.reset()
            return None

    def add_documents(self, chunks: List[Document], force_rebuild: bool = False) -> bool:
        if force_rebuild:
            self.reset()
        logger.info(f"Adding/updating {len(chunks)} chunks in Pinecone index '{self.index_name}'...")
        self._db.add_documents(chunks, batch_size=100)
        return True

    def reset(self) -> bool:
        logger.warning(f"Deleting Pinecone index '{self.index_name}'...")
        if self.pinecone_client.has_index(self.index_name):
            self.pinecone_client.delete_index(self.index_name)
        self._db = self._initialize_db()
        return True

class VectorStoreManager:
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config

    def get_vector_store(self) -> VectorStoreInterface:
        """
        Factory function that returns the appropriate vector store manager
        based on the application configuration.
        """
        provider = self.config.Database.provider
        logger.info(f"Vector store provider: {provider.value}")

        match provider: 
            case VectorStoreProvider.CHROMA: 
                return ChromaManager(self.config)
            case VectorStoreProvider.PINECONE: 
                return PineconeManager(self.config)
            case _: 
                raise ValueError(f"Unsupported vector store provider: {provider}")

    