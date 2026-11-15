from typing_extensions import TypedDict
from typing import Annotated,Sequence
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    original_question: str
    current_question: str
    documents: list[Document]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    selected_db:str
    no_of_retries:int
    max_retries:int
    next_step:str

    