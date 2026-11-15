from src.exception.exception import AgenticException
from src.logger.logger import logging
from src.agentic_rag.states.state import AgentState
from typing_extensions import Literal
from pydantic import BaseModel,Field
import sys
from langchain_core.prompts import PromptTemplate

class grade(BaseModel):
    "binary score for relevance"
    binary_score:str=Field(description="Binary score 'yes' or 'no' based on the relevance of the documents")

class Routers:
    def __init__(self,llm):
        self.llm=llm

    def grade_documents(self, state: AgentState) -> dict:
        try:
            grader_llm = self.llm.with_structured_output(grade, method="json_mode")

            prompt = PromptTemplate(
                            template="""You are a grader assessing relevance of a retrieved document to a user question.
                            Here is the retrieved document:

                            {context}

                            Here is the user question: {question}

                            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
                            Respond ONLY with a JSON object in exactly this format:
                            {{"binary_score": "yes"}} or {{"binary_score": "no"}}""",
                                        input_variables=["context", "question"],
                                    )

            chain = prompt | grader_llm
            retries = state["no_of_retries"]
            max_retries = state["max_retries"]

            result = chain.invoke({
                "question": state["current_question"],
                "context": state["documents"],
            })
            score = result.binary_score.strip().lower()

            if score.startswith("yes"):
                logging.info("Documents are relevant, routing to generate")
                return {"next_step": "generate", "no_of_retries": retries}

            if retries < max_retries:
                logging.info("Documents not relevant, routing to rewrite")
                return {"next_step": "rewrite", "no_of_retries": retries + 1}

            logging.info("Retries exhausted, routing to fallback")
            return {"next_step": "fall_back", "no_of_retries": retries}
        except Exception as e:
            logging.error(f"Exception {e} occurred while grading documents")
            raise AgenticException(e, sys)

    def route_next(self,state:AgentState):
        return state["next_step"]