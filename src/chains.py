from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from .session import SessionStore

from typing import Dict

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

class ChainFactory: 
    def __init__(self, llm, retrieve_context_f, format_context_f):
        self.llm = llm
        self.retrieve_context_f = retrieve_context_f       
        self.format_context_f = format_context_f

    def create_query_chain(self) -> Runnable:
        """Build LCEL chain for single queries without conversation history"""
        
        chain = (
            RunnablePassthrough()
            | PROMPTS["query"]
            | self.llm
            | StrOutputParser()
        )
        
        return chain

    def create_conversational_chain(self, session_store: SessionStore) -> Runnable:
        """ Build LCEL chain for chat with conversation history """
        
        # Build the chain with context retrieval
        def prepare_chat_input(inputs):
            question = inputs["question"]
            history = inputs.get("history", [])
            
            # Retrieve context based on current question
            rag_context = self.retrieve_context_f(question)
            formatted_context = self.format_context_f(rag_context)
            
            return {
                "context": formatted_context,
                "question": question,
                "history": history, 
                "rag_context": rag_context 
            }
        
        chat_chain = (
            RunnableLambda(prepare_chat_input)
            | RunnableParallel({
                "response": PROMPTS["chat"] | self.llm | StrOutputParser(),
                "rag_context": RunnableLambda(lambda x: x["rag_context"])   
            })
        )
        
        return RunnableWithMessageHistory(
            chat_chain,
            session_store.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            # Explicitly define the output key for the AI's response, to suppress the key error warning. 
            # The output is used for analysis for tracing solutions like langsmith
            output_messages_key="response",
        )