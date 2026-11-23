import sys
from src.logger.logger import logging
from langgraph.errors import NodeError
from langgraph.types import Command
from src.agentic_rag.states.state import AgentState

class AgenticException(Exception):
    def __init__(self, error_message, error_details: sys = None):
        super().__init__(str(error_message))
        self.error_message = error_message

        exc_tb = None
        if error_details is not None:
            _, _, exc_tb = error_details.exc_info()
        if exc_tb is None and isinstance(error_message, BaseException):
            exc_tb = error_message.__traceback__

        self.lineno = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )

    def agentnode_error(state,error:NodeError)-> Command:
        logging.error("Node [%s] failed: %s", error.node, error.error)

        return Command(
        goto="gracefull_fallback",
        update={"errors": state.get("errors", []) + [str(error.error)]},
    )