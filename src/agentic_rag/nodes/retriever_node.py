from src.logger.logger import logging
from src.exception.exception import AgenticException
from src.agentic_rag.states.state import AgentState
from src.agentic_rag.utils.query_utils.query_utils import query_utils
from src.agentic_rag.states.state import AgentState
from src.pinecone_setup.data import VECTOR_INDEXES,HYBRID_INDEXES
import sys

class Retriever:
    def __init__(self,query_utils):
        self.query_utils=query_utils
        

    def retriever_helper(self,state:AgentState,indexes):
        try:
            if state['selected_db']=="langchain-db":
                index_name=indexes["langchain-hybrid-index"]["name"]
                index_host=indexes["langchain-hybrid-index"]["host"]
            elif state['selected_db']=="langgraph-db":
                index_name=indexes["langgraph-hybrid-index"]["name"]
                index_host=indexes["langgraph-hybrid-index"]["host"]
            return index_name,index_host
        except Exception as e:
            logging.error("Error has occurred while selecting database",{e})
            raise AgenticException(e,sys)
        

    def retriever(self,state:AgentState,indexes):
        index_name,index_host=self.retriever_helper(indexes)
        try:
            query=state["current_question"]
            retrieved_docs=self.query_utils.query_hybrid_index(index_name,index_host,query)
            return retrieved_docs
        except Exception as e:
            logging.error("Error has occurred while retrieving data from pinecone",{e})
            raise AgenticException(e,sys)
        