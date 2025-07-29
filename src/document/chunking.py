from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from src.embedding import Embeddings
from src.config.config import RAGConfig
from src.config.constants import ChunkerType

import logging
logger = logging.getLogger(__name__)

# The Registry: Maps the enum to the corresponding class
CHUNKERS_REGISTRY = {
    ChunkerType.RECURSIVE_CHARACTER_TEXT_SPLITTER: RecursiveCharacterTextSplitter,
    ChunkerType.SEMANTIC_CHUNKER: SemanticChunker,
}

def get_chunker(config: RAGConfig, embeddings: Embeddings) -> 'TextSplitter':
    """
 and configure the text splitter based on the config.
    """
    chunker_type = config.DocProcessor.chunker_type
    chunker_class = CHUNKERS_REGISTRY.get(chunker_type)

    if not chunker_class:
        raise ValueError(f"Unknown chunker type: {chunker_type}")

    # Common parameters can be defined here
    chunker_params = {"add_start_index": True}  

    # Add specific parameters for each type
    if chunker_type == ChunkerType.RECURSIVE_CHARACTER_TEXT_SPLITTER:
        logger.info("Using Recursive Character Text Splitter.")
        chunker_params.update({
            "chunk_size": config.DocProcessor.chunk_size,
            "chunk_overlap": config.DocProcessor.chunk_overlap,
            "separators": ['\n\n', '\n', '.', '?', '!', ' ', ''],
        })
    elif chunker_type == ChunkerType.SEMANTIC_CHUNKER:
        logger.info("Using Semantic Chunker.")
        chunker_params.update({
            "embeddings": embeddings.get_embedding_model(),
            "breakpoint_threshold_type": "standard_deviation",
        })

    return chunker_class(**chunker_params)