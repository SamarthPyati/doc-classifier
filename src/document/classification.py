from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from src.config import RAGConfig
from src.llm import LLMFactory

import logging
logger = logging.getLogger(__name__)

# Define the desired structured output for the classifier
class DocumentCategory(BaseModel):
    category: str = Field(description="The single most appropriate category for the document.")

class DocumentClassifier:
    """ Uses an LLM to classify document content into predefined categories """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm_factory = LLMFactory(config) 
        self.llm = self.llm_factory.get_llm()
        self.parser = PydanticOutputParser(pydantic_object=DocumentCategory) # type: ignore
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        """ Creates the prompt template for the classification task """
        categories = self.config.DocProcessor.classification_categories
        
        prompt_template = """
        You are an expert document classifier. Your task is to analyze the following text content and assign it to ONE of the predefined categories.

        Available Categories:
        {categories}

        Review the text sample below and determine the single best category.
        {format_instructions}

        Text Sample:
        ---
        {text_sample}
        ---
        """
        return ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={
                "categories": ", ".join(categories),
                "format_instructions": self.parser.get_format_instructions(),
            },
        )

    def classify_content(self, content_sample: str) -> str:
        """
        Classifies the given text sample and returns the category name 
        """
        if not content_sample:
            return "General"
            
        try:
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({"text_sample": content_sample})
            return result.category
        except Exception as e:
            logger.error(f"Error in document classification: {e}", exc_info=True)
            # Fallback to a default category on error
            return "General"
