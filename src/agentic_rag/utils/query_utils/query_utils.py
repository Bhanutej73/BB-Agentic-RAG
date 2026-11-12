import os,sys,types,ast

# Stub for the ChatVertexAI import that ragas 0.4.x needs but langchain-community 0.4.x no longer ships
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:  # placeholder, never used
    pass
_stub.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub
from functools import partial
import pandas as pd
from src.pinecone_setup.data import EVALUATION_DATASET_PATH, LANGCHAIN_EVALUATION_DATASET_PATH, LANGGRAPH_EVALUATION_DATASET_PATH
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from src.logger.logger import logging
from src.exception.exception import AgenticException
from src.pinecone_setup.data import VECTOR_INDEXES,HYBRID_INDEXES
from langchain_core.documents import Document
from src.agentic_rag.utils.eval_utils.eval_utils import EvaluationUtils
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper


load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})

langchain_api_key=os.getenv("GROQ_API_KEY")
eval_llm=ChatGroq(api_key=langchain_api_key, model="qwen/qwen3.8-27b")
eval_llm_wrapper=LangchainLLMWrapper(eval_llm)

class query_utils:
    def __init__(self, embeddings, pc_client, top_k:int=10):
        self.embeddings=embeddings
        self.top_k=top_k
        self.pc_client=pc_client

    def reciprocal_rank_fusion(self, rankings, k=60):
        scores = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return scores

    def to_documents(self, results, fused):
        """Convert Pinecone matches into LangChain Documents."""
        docs = []
        for m in results:
            # Full payload of the match (needs include_fields=["*"] on the searches)
            data = m.to_dict() if hasattr(m, "to_dict") else {}

            title = data.get("title") or getattr(m, "title", "") or ""
            description = data.get("description") or getattr(m, "description", "") or ""
            content = data.get("page_content") or getattr(m, "page_content", "") or ""

            # Everything else that came back (source URL, etc.) becomes metadata
            skip = {"_id", "_score", "title", "description", "page_content", "embedding"}
            extra = {k: v for k, v in data.items() if k not in skip}

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": m._id,
                        "title": title,
                        "description": description,
                        "rrf_score": fused.get(m._id),
                        "search_score": getattr(m, "_score", None),
                        **extra,
                    },
                )
            )
        return docs

    def query_vector_db(self,index_name:str, host:str, query:str):
        try:
            index=self.pc_client.index(index_name, host=host,grpc=True)
            embedded_query=self.embeddings.embed_query(query)
            results=index.query(vector=embedded_query, top_k=self.top_k, include_metadata=True)
            return [
                Document(
                    page_content=(m.metadata or {}).get("page_content", ""),
                    metadata={
                        "id":m.id,
                        "search_score":m.score,
                        **{k: v for k,v in (m.metadata or {}).items() if k!="page_content"},
                    },
                )
                for m in results.matches
            ]
        except Exception as e:
            logging.error(f"Error occurred while querying the vector database: {str(e)}")
            raise AgenticException(f"Error occurred while querying the vector database: {str(e)}",sys)

    def query_hybrid_index(self,index_name:str, host:str, query:str):
        try:
            #self.index=pc.index(self.index_name, host=self.host)
            index=self.pc_client.index(index_name, host=host)
            embedded_query=self.embeddings.embed_query(query)
            #vector part
            dense_hits=index.documents.search(
                namespace="__default__", top_k=self.top_k,
                score_by=[{"type":"dense_vector","field":"embedding","values":embedded_query}],
                include_fields=["*"]
            ).matches

            #BM25 part
            bm25_hits=index.documents.search(
                namespace="__default__", top_k=self.top_k,
                score_by=[
                    {"type": "text", "field": "title", "query": query},
                    {"type": "text", "field": "description", "query": query},
                    {"type": "text", "field": "page_content", "query": query},
                ],
                include_fields=["*"]
            ).matches

            #RRF
            fused=self.reciprocal_rank_fusion([
                [m._id for m in bm25_hits],
                [m._id for m in dense_hits],]
            )

            by_id = {m._id: m for m in bm25_hits}
            for m in dense_hits:
                by_id.setdefault(m._id, m)

            results = [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)[:self.top_k]]
            return self.to_documents(results, fused)
        except Exception as e:
            logging.error(f"Error occurred while querying the hybrid index: {str(e)}")
            raise AgenticException(f"Error occurred while querying the hybrid index: {str(e)}",sys)


if __name__=="__main__":
    try:
        query="How to install Langchain?"
        path=LANGCHAIN_EVALUATION_DATASET_PATH
        eval_dataset=pd.read_csv(path)
        eval_dataset["reference_contexts"]=eval_dataset["reference_contexts"].apply(ast.literal_eval)
        eval_dataset["reference"] = eval_dataset["reference"].fillna("")
        index_name=HYBRID_INDEXES["langchain-hybrid-index"]["name"]
        host_name=HYBRID_INDEXES["langchain-hybrid-index"]["host"]
        eval_utils=EvaluationUtils(eval_llm_wrapper) 
        query_util=query_utils(embeddings, pc_client=pc)
        query_index=partial(query_util.query_hybrid_index,index_name=index_name,host=host_name)
        # results=query_util.query_hybrid_index(index_name, host_name, query)
        # print(results)
        dataset=eval_utils.make_dataset(eval_dataset,query_index=query_index)
        eval_results=eval_utils.retrieval_evaluation(dataset)
        print(eval_results)
    except Exception as e:
        logging.error(f"Error occurred in main: {str(e)}")
        raise AgenticException(f"Error occurred in main: {str(e)}",sys)