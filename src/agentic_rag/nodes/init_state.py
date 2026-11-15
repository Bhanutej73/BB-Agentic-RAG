from src.agentic_rag.states.state import AgentState

class InitState:

    def __init__(self):
        pass

    def __call__(self, state: AgentState):
        question = state["messages"][-1].content

        return {
            "original_question": question,
            "current_question": question,
            "documents": [],
            "selected_db": "",
            "no_of_retries": 0,
            "max_retries": 5,
            "next_step": "",
        }