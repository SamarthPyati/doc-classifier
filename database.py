from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.config import Settings

from config import RAGConfig, DEFAULT_RAG_CONFIG, logger

import os
from pathlib import Path
from typing import List 
from dotenv import load_dotenv

load_dotenv()

# TODO: Migrate the vector database to Pinecone/Weaviate DB for better scalability
class VectorStoreManager: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.config = config 
        # TODO: Use better embedding model preferrably AWS Bedrock or OpenAI
        self.embedding_function = None
        if True: 
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=self.config.Database.embedding_model, 
                model_kwargs = {'device': 'mps'},           # MacOS
                multi_process=True
            )
        else: 
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-exp-03-07"
            )

        self.database_path = os.path.abspath(self.config.Database.database_path)

    def create_vector_store(self, documents: List[Document], overwrite: bool = False) -> bool:
        """ Create a vector database from the documents """
        # TODO: Integrate metadata checking for indexing only the updated file
        try: 
            if overwrite or Path(self.database_path).exists(): 
                import shutil 
                shutil.rmtree(self.database_path)
                logger.info("Removed existing database ...")

            if not documents: 
                logger.warning("No documents to index")
                
            Chroma.from_documents(
                documents=documents, 
                embedding=self.embedding_function, 
                persist_directory=self.config.Database.database_path,
                client_settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Saved {len(documents)} chunks to {self.database_path}")

        except Exception as e: 
            logger.error(f"Error in creating vector store: {e}")
            raise

    def load_vector_store(self) -> Chroma:
        """ Load existing vector store """
        try:
            if not Path(self.database_path).exists():
                raise FileNotFoundError(f"Vector store not found at {self.database_path}")
            
            db = Chroma(
                persist_directory=self.database_path,
                embedding_function=self.embedding_function,
                client_settings=Settings(anonymized_telemetry=False)
            )

            logger.info(f"Loaded vector store from {self.database_path}")
            return db
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def update_database(self) -> Chroma: 
        None

    def similarity_search(self, query: str, db: Chroma) -> List[tuple[Document, float]]: 
        """ Perform similarity search with a query and return all the relevant files """
        try: 
            results = db.similarity_search_with_relevance_scores(
                query, 
                k = self.config.Database.max_results, 
                score_threshold=self.config.Database.similarity_threshold
            )

            results.sort(key=lambda x : x[1], reverse=True)
            
            logger.info(f"Found {len(results)} relevant document for query: \"{query}\"")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise   