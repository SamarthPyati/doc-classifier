from typing import Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, BasePromptTemplate

PROMPTS: Dict[str, BasePromptTemplate] = {
    "query": PromptTemplate( 
        input_variables=["context", "question"],
        template="""You are an expert AI assistant that provides accurate, comprehensive and concise answers based on the given context.
            Context Information:
            {context}
            
            Question: {question}
            
            Instructions:
            - Answer using ONLY the provided context
            - Be comprehensive yet concise
            - If context is insufficient, state this clearly, do not make up any answers by yourself
            - Cite specific sources when mentioning details
            - Provide structured, actionable information when possible
            
            Answer:"""
    ), 

    "chat": ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI assistant engaged in a conversation. Use the provided context and conversation history to give accurate, helpful responses.

            Context Information:
            {context}

            Instructions:
            - Use the context and conversation history to provide relevant answers
            - Be conversational and natural while staying accurate
            - Reference previous parts of the conversation when relevant
            - If the question relates to earlier topics, acknowledge that connection
            - If context is insufficient for a complete answer, ask clarifying questions
            - Maintain consistency with previous responses
            - Always cite sources when providing specific information"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
    ])
}