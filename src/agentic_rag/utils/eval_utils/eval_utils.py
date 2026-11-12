import sys, types
import os
import ast
# Stub for the ChatVertexAI import that ragas 0.4.x needs but langchain-community 0.4.x no longer ships
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:  # placeholder, never used
    pass
_stub.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub

from langchain_core.documents import Document
from src.logger.logger import logging
from src.exception.exception import AgenticException
from ragas import EvaluationDataset
import pandas as pd
from src.pinecone_setup.data import EVALUATION_DATASET_PATH, LANGCHAIN_EVALUATION_DATASET_PATH, LANGGRAPH_EVALUATION_DATASET_PATH
from ragas import evaluate
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
load_dotenv(override=True)
langchain_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(api_key=langchain_api_key, model="qwen/qwen3.8-27b")
llm_wrapper=LangchainLLMWrapper(llm)

from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig
import dagshub


class EvaluationUtils:
    def __init__(self,llm_wrapper):
        self.llm_wrapper=llm_wrapper
        

    def load_golden_dataset(self, dataset_path:str, only_keep:bool=True)->EvaluationDataset:
        try:
            df=pd.read_csv(dataset_path)

            if only_keep and "keep" in df.columns:
                df=df[df["keep".astype(str).str.lower().isin(["true","1","yes","y"])]]

            df["reference_contexts"]=df["reference_contexts"].apply(
                    lambda v: ast.literal_eval(v) if isinstance(v,str) else v
            )

            ragas_df=df[["user_input", "reference", "reference_contexts"]]
            dataset=EvaluationDataset.from_list(ragas_df.to_dict(orient="records"))

            meta_df=df.drop(coulmns=["user_input", "reference", "reference_contexts"])
            return dataset,meta_df            
        except Exception as e:
            logging.error(f"Error loading dataset from {dataset_path}: {str(e)}")
            raise AgenticException(e, sys)

    def make_dataset(self, golden_dataset:pd.DataFrame,query_index):
        dataset=[]
        try:
            for user_input, reference_contexts, reference in zip(golden_dataset["user_input"], golden_dataset["reference_contexts"], golden_dataset["reference"]):
                retrieved_contexts=query_index(query=user_input)
                page_context=[doc.page_content for doc in retrieved_contexts]
                dataset.append(
                    {
                        "user_input":user_input,
                        "reference_contexts":reference_contexts,
                        "retrieved_contexts":page_context,
                        "reference":reference
                        
                    }
                )
            evaluation_dataset=EvaluationDataset.from_list(dataset)
            return evaluation_dataset
        except Exception as e:
            logging.error(f"Error creating evaluation dataset: {str(e)}")
            raise AgenticException(e, sys)
    
    def retrieval_evaluation(self,evaluation_dataset:EvaluationDataset):
        try:
            result=evaluate(dataset=evaluation_dataset,metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],llm=llm_wrapper,
                            run_config=RunConfig(timeout=600,max_workers=1,max_retries=10, max_wait=60),
                            raise_exceptions=False,show_progress=True)
            return result
        except Exception as e:
            logging.error(f"Error occurred during retrieval evaluation: {str(e)}")
            raise AgenticException(e, sys)

    def generation_evaluation(self): 
        try:
            pass
        except Exception as e:
            logging.error(f"Error occurred during generation evaluation: {str(e)}")
            raise AgenticException(e, sys)