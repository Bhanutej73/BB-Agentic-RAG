from src.pinecone_setup.data import LANGCHAIN_DATA_SOURCE,LANGGRAPH_DATA_SOURCE
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

loader = WebBaseLoader()

langchain_docs=LANGCHAIN_DATA_SOURCE
langgraph_docs=LANGGRAPH_DATA_SOURCE

def extract_langchain_data():
    langchain_documents=[]
    for url in langchain_docs:
        langchain_documents=WebBaseLoader(url).load()
    return langchain_documents