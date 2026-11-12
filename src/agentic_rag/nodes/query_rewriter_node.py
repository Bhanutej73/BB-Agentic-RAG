from src.exception.exception import AgenticException
from src.logger.logger import logging
from src.agentic_rag.states.state import AgentState
from langchain_core.messages import HumanMessage,AIMessage
import sys

class query_rewriter:
    def __init__(self,llm):
        self.llm=llm

    def __call__(self,state:AgentState):
        try:
            """
                Transform the query to produce a better question.
            
                Args:
                    state(messages): The current state
            
                Returns:
                    dict: The updated state with a re-phrased question
            """
            original_question=state['original_question']
            msg=[
                HumanMessage(
                    content=f"""\n
                    Look at the input and try to reason about the underlying semantic intent / meaning. \n
                    Here is the initial question:
                    \n ------\n
                    {original_question}
                    \n ------- \n
                    Formulate an improved question: """
                )
            ]
            rewritten_question=self.llm.invoke(msg)
            return{
                    "current_question":rewritten_question.content,
                    "messages": AIMessage(content=rewritten_question.content)
            }
        except Exception as e:
            raise AgenticException(e,sys)
        
        