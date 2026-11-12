from src.agentic_rag.states.state import AgentState
from src.exception.exception import AgenticException
from src.logger.logger import logging
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import sys

class Generator:
    def __init__(self,llm):
        self.llm=llm        

    def preprocess_context(self,context:Document):
        try:
            preprocessed_context="".join(
                doc.page_content  for doc in context
            )
            logging.info("context pre-processed successfully")
            return preprocessed_context
        except Exception as e:
            logging.error("Error occurred while pre-processing the context")
            raise AgenticException(e,sys)

    def generator(self,state:AgentState):

        """
            Generate answer
    
            Args:
                state(messages); The current state

            Returns:
                dict; The updated message
        """
               
        prompt="""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """    
        chain=prompt | self.llm | StrOutputParser()

        try:
            question=state["current_question"]
            context=state["documents"]
            preprocessed_context=self.preprocess_context(context=context)
            response=chain.invoke({"context":preprocessed_context,"question":question})
            logging.info("Response generated successfully")
            return {"messages":[AIMessage(content=response)]}
        except Exception as e:
            logging.error("Error occurred while generating response")
            raise AgenticException(e,sys)                                                      