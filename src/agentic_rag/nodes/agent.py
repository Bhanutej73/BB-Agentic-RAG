from src.logger.logger import logging
from src.exception.exception import AgenticException
from src.agentic_rag.states.state import AgentState
from typing_extensions import Literal
from pydantic import BaseModel,Field
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

class Route(BaseModel):
     """Choose which knowledge source should be searched for the question."""
     retriever: Literal["langchain-db", "langgraph-db"] = Field(
        description="The documentation source best suited to answer the question."  
    )

class Agent:
    def __init__(self,llm):
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
        response = self.llm.with_structured_output(Route, method="json_mode").invoke([
        SystemMessage(
            content="""You are a routing assistant. Do NOT answer the user's question.
                    Decide which documentation source should be searched for the question.

                    Respond ONLY with a valid JSON object in exactly this format:
                    {"retriever": "<one of: langchain-db, langgraph-db>"}
                    """
        ),
        HumanMessage(content=question)
    ])

        return {
            "selected_db": response.retriever,
            "messages": [
                AIMessage(content=f"Selected DB: {response.retriever}")
            ]
        }