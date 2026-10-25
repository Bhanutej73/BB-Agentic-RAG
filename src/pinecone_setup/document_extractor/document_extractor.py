from src.pinecone_setup.data import LANGCHAIN_DATA_SOURCE,LANGGRAPH_DATA_SOURCE
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from src.logger.logger import logging
from src.exception.exception import AgenticException
logging.basicConfig(level=logging.INFO)
import sys
loader = WebBaseLoader()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

class DocumentExtractor:
    def __init__(self,langchain_docs,langgraph_docs,text_spitter):
        self.langchain_docs=langchain_docs
        self.langgraph_docs=langgraph_docs
        self.langchain_documents=[]
        self.langgraph_documents=[]
        self.text_splitter=text_spitter

    def split_text(self, documents):
        return self.text_splitter.split_documents(documents)

    def extract_langchain_data(self):
        docs=[]
        for url in self.langchain_docs:
            try:
                docs.extend(WebBaseLoader(url).load())
                for doc in docs:
                    self.langchain_documents.extend(self.split_text([doc]))
            except Exception as e:
                logging.error(f"Error occurred while extracting data from {url}: {str(e)}")
                raise AgenticException(e,sys)
        return self.langchain_documents

    def extract_langgraph_data(self):
        docs=[]
        for url in self.langgraph_docs:
            try:
                docs.extend(WebBaseLoader(url).load())
                for doc in docs:
                    self.langgraph_documents.extend(self.split_text([doc]))
            except Exception as e:
                logging.error(f"Error occurred while extracting data from {url}: {str(e)}")
                raise AgenticException(e,sys)
        return self.langgraph_documents
    
    

if __name__=="__main__":
    doc_extractor=DocumentExtractor(LANGCHAIN_DATA_SOURCE,LANGGRAPH_DATA_SOURCE,text_splitter)
    langchain_docs=doc_extractor.extract_langchain_data()
    langgraph_docs=doc_extractor.extract_langgraph_data()
    print(langchain_docs)
    print(langgraph_docs)