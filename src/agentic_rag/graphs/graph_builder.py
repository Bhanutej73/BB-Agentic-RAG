from src.exception.exception import AgenticException
from src.logger.logger import logging
from src.agentic_rag.states.state import AgentState
from src.agentic_rag.nodes.agent import Agent
from src.agentic_rag.nodes.generator_node import Generator
from src.agentic_rag.nodes.query_rewriter_node import query_rewriter
from src.agentic_rag.nodes.retriever_node import Retriever
from src.agentic_rag.nodes.routers import Routers
from langgraph.graph import StateGraph,START,END
from src.pinecone_setup.data import VECTOR_INDEXES,HYBRID_INDEXES
from IPython.display import Image,display
from src.agentic_rag.llm.groqllm import GroqLLM
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from src.agentic_rag.utils.query_utils.query_utils import query_utils
from src.agentic_rag.nodes.init_state import InitState
from src.agentic_rag.nodes.gracefull_fallback import Fallback
import os
from dotenv import load_dotenv
load_dotenv()


class Graph:
    def __init__(self,llm,Agent:Agent,Generator:Generator,query_rewriter:query_rewriter,Retriever:Retriever,Routers:Routers,InitState:InitState,Fallback:Fallback):
        self.llm=llm
        self.graph=StateGraph(AgentState)
        self.agent=Agent(llm)
        self.Generator=Generator(llm)
        self.query_rewriter=query_rewriter(llm)
        self.retriever=Retriever
        self.router=Routers(llm)
        self.init_state=InitState()
        self.fallback=Fallback()

    def __call__(self):
        #defining nodes
        self.graph.add_node("init_state",self.init_state)
        self.graph.add_node("agent",self.agent)
        self.graph.add_node("retriever",self.retriever)
        self.graph.add_node("generator",self.Generator.generator)
        self.graph.add_node("rewriter",self.query_rewriter)
        self.graph.add_node("document_grader",self.router.grade_documents)
        self.graph.add_node("gracefull_fallback",self.fallback)
        #self.graph.add_node("db_router",self.router.route_retriever())

        #constructing the graph with edges
        self.graph.add_edge(START,"init_state")
        self.graph.add_edge("init_state","agent")
        self.graph.add_edge("agent","retriever")
        self.graph.add_edge("retriever","document_grader")
        self.graph.add_conditional_edges("document_grader",
                                         self.router.route_next,{
                                             "rewrite":"rewriter",
                                             "generate":"generator",
                                             "fall_back":"gracefull_fallback"
                                         })
        self.graph.add_edge("rewriter","agent")
        self.graph.add_edge("gracefull_fallback",END)
        self.graph.add_edge("generator",END)
        return self.graph.compile()
        

if __name__=="__main__":
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
    llm=GroqLLM().get_llm()
    query_util=query_utils(embeddings, pc_client=pc)
    retriever=Retriever(query_utils=query_util)
    graph_builder=Graph(llm,Agent,Generator,query_rewriter,retriever,Routers,InitState,Fallback)
    compiled_graph = graph_builder()
    png = compiled_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png)
    # response=compiled_graph.invoke({"messages":"What is machine learning"})
    # print(response)