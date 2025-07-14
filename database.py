from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import RAGConfig, DEFAULT_RAG_CONFIG 

import os 
from typing import List 
import logging

logging.basicConfig(
    filename="events.log", 
    level=logging.INFO, 
)
logger = logging.getLogger(__name__)

# TODO: Shift the vector database to Qdrant for better scalability
class VectorStoreManager: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config 
        # TODO: Use better embedding model preferrably AWS Bedrock or OpenAI
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.config.Database.embedding_model
        )
        self.database_path = os.path.abspath(self.config.Database.database_path)

    def create_vector_store(self, documents: List[Document]) -> bool:
        """ Create a vector database from the documents """
        try: 
            # TODO: Integrate metadata checking for indexing only the updated file
            Chroma.from_documents(documents=documents, 
                                  embedding=self.embedding_function, 
                                  persist_directory=self.config.Database.database_path)
            
            logger.info(f"Saved {len(documents)} chunks to {self.database_path}")

        except Exception as e: 
            logger.error(f"Error in creating vector store: {e}")
            raise

    def load_vector_store(self) -> Chroma:
        """ Load existing vector store """
        try:
            if not os.path.exists(self.database_path):
                raise FileNotFoundError(f"Vector store not found at {self.database_path}")
            
            db = Chroma(
                persist_directory=self.database_path,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded vector store from {self.database_path}")
            return db
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def similarity_search(self, query: str, db: Chroma) -> List[tuple[Document, float]]: 
        """ Perform similarity search with a query and return all the relevant files """
        try: 
            results = db.similarity_search_with_relevance_scores(
                query, 
                k=self.config.Database.max_results, 
            )

            # Filter results with threshold
            filtered_results = [(doc, rank) for (doc, rank) in results if rank >= self.config.Database.similarity_threshold]
            logger.info(f"Found {len(filtered_results)} relevant document for query: \"{query}\"")
            
            return filtered_results
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise   