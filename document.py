from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.evaluation.schema import PairwiseStringEvaluator
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Union, Iterator
import os 

from config import RAGConfig, DEFAULT_RAG_CONFIG, logger

class DocumentProcessor: 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG) -> None:
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.DocProcessor.chunk_size, 
            chunk_overlap=config.DocProcessor.chunk_overlap, 
            separators=['\n\n', '\n', '', ' '], 
            add_start_index=True,   # For getting offset of file at split
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self, lazy_load: bool = False) -> Union[List[Document], Iterator[Document]]: 
        """ Load documents from corpus """
        documents = []

        # TODO: Remove redundancy and make functions for each FileType
        try:
            for dirpath, _, filenames in os.walk(self.config.corpus_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)

                    try:
                        if filename.lower().endswith(".pdf"):
                            pdf_loader = PyMuPDFLoader(filepath)
                            docs = pdf_loader.load()
                            documents.extend(docs)
                            logger.info(f"Loaded {len(docs)} pages from PDF file: {filename}")

                        elif filename.lower().endswith((".txt", ".md", ".xhtml", ".html")):
                            loader = UnstructuredFileLoader(filepath)
                            docs = loader.load()
                            # logger.info(f"Loaded {len(docs)} docs from file: {filename}")
                            documents.extend(docs)

                        else:
                            logger.warning(f"Unsupported file format: {filename}")

                    except Exception as file_err:
                        logger.error(f"Failed to load {filename}: {file_err}")

            logger.info(f"Total documents loaded: {len(documents)}")    
            return documents
        except Exception as e: 
            logger.error("Error loading documents: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]: 
        """ Split the documents into chunks which are smaller constituent documents """
        try: 
            result = self.text_splitter.split_documents(documents=documents)
            return result
        except Exception as e: 
            logger.error("Error splitting documents: {e}")
            raise


class CosineEmbeddingEvaluator(PairwiseStringEvaluator):
    """ Custom evaluator for computing cosine similarity between embeddings """
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn

    def _evaluate_string_pairs(self, prediction, prediction_b, **kwargs):
        """ Evaluate similarity between two strings """
        try:
            vec_a = self.embedding_fn.embed_query(prediction)
            vec_b = self.embedding_fn.embed_query(prediction_b)
            sim = cosine_similarity([vec_a], [vec_b])[0][0]
            return {
                "score": sim,
                "explanation": f"Cosine similarity: {sim:.4f}"
            }
        except Exception as e:
            logger.error(f"Error evaluating string pairs: {e}")
            return {"score": 0.0, "explanation": f"Error: {e}"}
