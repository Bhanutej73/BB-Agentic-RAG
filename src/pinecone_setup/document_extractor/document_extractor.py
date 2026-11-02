from src.pinecone_setup.data import LANGCHAIN_DATA_SOURCE,LANGGRAPH_DATA_SOURCE
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from src.logger.logger import logging
from src.exception.exception import AgenticException
logging.basicConfig(level=logging.INFO)
import sys
loader = WebBaseLoader()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

import re
import sys
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

class DocumentExtractor:
    def __init__(self, langchain_docs, langgraph_docs, text_spitter):
        self.langchain_docs = langchain_docs
        self.langgraph_docs = langgraph_docs
        self.text_splitter = text_spitter

    def split_text(self, documents):
        return self.text_splitter.split_documents(documents)

    def _load_pages(self, urls):
        pages = []
        for url in urls:
            try:
                pages.extend(WebBaseLoader(url).load())
            except Exception as e:
                logging.error(f"Error occurred while extracting data from {url}: {str(e)}")
                raise AgenticException(e, sys)
        return pages

    # ----- split chunks, used by ingestion -----
    def extract_langchain_data(self):
        chunks = self.split_text(self._load_pages(self.langchain_docs))
        logging.info(f"Extracted and split LangChain documents: {len(chunks)} chunks.")
        return chunks

    def extract_langgraph_data(self):
        chunks = self.split_text(self._load_pages(self.langgraph_docs))
        logging.info(f"Extracted and split LangGraph documents: {len(chunks)} chunks.")
        return chunks

    # ----- whole pages, used by ragas test set generation -----
    def extract_whole_docs(self, urls, max_docs=None, max_chars=3000, min_chars=500):
        cleaned = []
        for d in self._load_pages(urls):
            text = re.sub(r"\n\s*\n+", "\n\n", d.page_content).strip()
            if len(text) >= min_chars:
                cleaned.append(Document(page_content=text[:max_chars], metadata=d.metadata))
        if max_docs:
            cleaned = cleaned[:max_docs]
        total_chars = sum(len(d.page_content) for d in cleaned)
        logging.info(f"{len(cleaned)} docs | {total_chars} chars | ~{total_chars // 4} tokens per pass")
        return cleaned

    def extract_langchain_data_without_splitting(self, **kwargs):
        return self.extract_whole_docs(self.langchain_docs, **kwargs)

    def extract_langgraph_data_without_splitting(self, **kwargs):
        return self.extract_whole_docs(self.langgraph_docs, **kwargs)

if __name__=="__main__":
    doc_extractor=DocumentExtractor(LANGCHAIN_DATA_SOURCE,LANGGRAPH_DATA_SOURCE,text_splitter)
    langchain_docs=doc_extractor.extract_langchain_data()
    langgraph_docs=doc_extractor.extract_langgraph_data()
    print(langchain_docs)
    print(langgraph_docs)