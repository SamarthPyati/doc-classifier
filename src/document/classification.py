from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from src.config import RAGConfig, DEFAULT_RAG_CONFIG
from src.llm import LLMFactory
from src.constants import ClassificationMethod

from abc import ABC, abstractmethod
from typing import Dict

import logging
logger = logging.getLogger(__name__)

class DocumentClassifierInterface(ABC): 
    """ Base class for document classifiers """
    @abstractmethod
    async def classify(self, content_sample: str) -> str: 
        """ Classify content and return the category as str """
        pass

class KeywordClassifier(DocumentClassifierInterface): 
    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG): 
        self.keywords = config.DocProcessor.classification_keywords

    async def classify(self, content_sample: str): 
        if not content_sample: 
            return "GENERAL"
         
        content = content_sample.lower()
        scores: Dict[str, int] = {category : 0 for category, _ in self.keywords.items()}
        for category, keywords in self.keywords.items(): 
            for keyword in keywords: 
                scores[category] += content.count(keyword.lower())

        best_category = max(scores, key=scores.get)

        if all(score == 0 for _, score in scores.items()): 
            return "GENERAL"

        return best_category
    
class LLMClassifier(DocumentClassifierInterface):
    """ Uses an LLM to classify document content into predefined categories """
    class DocumentCategory(BaseModel):
        category: str = Field(description="The single most appropriate category for the document.")

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm_factory = LLMFactory(config) 
        self.llm = self.llm_factory.get_llm()
        self.parser = PydanticOutputParser(pydantic_object=LLMClassifier.DocumentCategory) 
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        """ Creates a prompt template for the classification task """
        categories = self.config.DocProcessor.classification_categories
        
        prompt_template = """
        You are an expert document classifier. Your task is to analyze the text sample provided and assign it to ONE of the following categories.
        You must respond in the requested JSON format.

        Available Categories:
        {categories}

        Text Sample:
        ---
        {text_sample}
        ---

        Based on the text, what is the single best category?

        {format_instructions}
        """
        return ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={
                "categories": ", ".join(categories),
                "format_instructions": self.parser.get_format_instructions(),
            },
        )

    async def classify(self, text_sample: str) -> str:
        """
        Classifies the given text sample and returns the category name 
        """
        if not text_sample:
            return "GENERAL"
            
        try:
            chain = self.prompt | self.llm | self.parser
            result = await chain.invoke({"text_sample": text_sample})
            return result.category  
        except Exception as e:
            logger.error(f"Error in document classification: {e}")
            # Fallback to a default category on error
            return "GENERAL"


def get_classifier(config: RAGConfig) -> DocumentClassifierInterface:
    """
    Factory function that returns the appropriate classifier instance
    based on the application configuration.
    """
    method = config.DocProcessor.classification_method
    logger.info(f"Selected classification method: {method.value}")

    if method == ClassificationMethod.KEYWORD:
        return KeywordClassifier(config)
    elif method == ClassificationMethod.LLM:
        return LLMClassifier(config)
    else:
        raise ValueError(f"Unsupported classification method: {method}")
