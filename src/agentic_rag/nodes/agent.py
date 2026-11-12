from src.logger.logger import logging
from src.exception.exception import AgenticException
from src.agentic_rag.states.state import AgentState
from typing_extensions import Literal
from pydantic import BaseModel,Field
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

class Route(BaseModel):
    retriever:Literal["langchain-db", "langgraph-db", "END"]

class Agent:
    def __init__(self,llm):
        retriever:Literal["langchain-db", "langgraph-db", "END"]
        self.llm=llm

    def __call__(self,state:AgentState)->dict:

        """
            Routes to the appropriate retriever tool based on the question in the state

            Args:
                state(messages): The current state
        
            Returns:
                dict: The updated state with the agent response appended to the router.
        """

        question=state["current_question"]
        response = self.llm.with_structured_output(Route).invoke([
        SystemMessage(
            content=f"You are an assistant for routing to the respective retriever based on the {question}, if the question is not related to the vector databases, route it to END"
        ),
        HumanMessage(content=question)
    ])

        return {
            "selected_db": response.retriever,
            "messages": [
                AIMessage(content=f"Selected DB: {response.retriever}")
            ]
        }