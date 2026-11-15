from langchain_core.messages import AIMessage
from src.agentic_rag.states.state import AgentState

class Fallback:
    def __call__(self, state: AgentState):
        return {
            "messages": [
                AIMessage(
                    content="Sorry, I couldn't find relevant information in the documentation to answer that question."
                )
            ]
        }