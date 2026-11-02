from src.exception.exception import AgenticException
from src.logger.logger import logging
from src.pinecone_setup.data import VECTOR_INDEXES,HYBRID_INDEXES
from src.pinecone_setup.document_extractor.document_extractor import DocumentExtractor
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
import os,sys
from src.pinecone_setup.data import LANGCHAIN_DATA_SOURCE,LANGGRAPH_DATA_SOURCE
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

import hashlib

RESERVED = {"title", "description", "page_content", "embedding"}

class DataIngestion:
    def __init__(self, document_extractor: DocumentExtractor, embeddings: HuggingFaceEmbeddings, vector_indexes=VECTOR_INDEXES, hybrid_indexes=HYBRID_INDEXES):
        self.document_extractor = document_extractor
        self.embeddings = embeddings
        self.vector_indexes = vector_indexes
        self.hybrid_indexes = hybrid_indexes
        
    def _rollback(self, index, ids, namespace, chunk_size=1000):
            try:
                for start in range(0,len(ids), chunk_size):
                    index.delete(ids=ids[start:start + chunk_size], namespace=namespace)
                logging.info(f"Rollback successful for namespace: {namespace}")
                logging.info(f"IDs of rolled back records: {ids}")
            except Exception as e:
                logging.error(f"Error occurred during rollback: {str(e)}")
                raise AgenticException(e, sys)

    @staticmethod
    def _make_id(doc, i: int) -> str:
        raw = f"{doc.metadata.get('source', '')}|{i}|{doc.page_content}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_metadata(meta: dict, hybrid: bool = False) -> dict:
        cleaned = {}
        for key, value in meta.items():
            if value is None:
                continue
            if hybrid and (key in RESERVED or key.startswith(("_", "$"))):
                continue   # reserved or schema-owned names can't be used as metadata
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
            
    def generate_langchain_embeddings(self):
        try:
            docs_to_embedd=self.document_extractor.extract_langchain_data()
            text=[doc.page_content for doc in docs_to_embedd]
            # self.sparse_embeddings.fit(text)
            # sparse_vectors=self.sparse_embeddings.encode_documents(text)
            embedded_docs=self.embeddings.embed_documents(text)
            logging.info("Successfully generated embeddings for LangChain documents.")
            return docs_to_embedd,embedded_docs
        except Exception as e:
            logging.error(f"Error occurred while generating embeddings: {str(e)}")
            raise AgenticException(e,sys)

    def generate_langgraph_embeddings(self):
        try:
            docs_to_embedd=self.document_extractor.extract_langgraph_data()
            text=[doc.page_content for doc in docs_to_embedd]
            # self.sparse_embeddings.fit(text)
            #sparse_vectors=self.sparse_embeddings.encode_documents(text)
            embedded_docs=self.embeddings.embed_documents(text)
            logging.info("Successfully generated embeddings for LangGraph documents.")
            return docs_to_embedd,embedded_docs
        except Exception as e:
            logging.error(f"Error occurred while generating embeddings: {str(e)}")
            raise AgenticException(e,sys)

    def upload_to_classic_vector_db(self,index_name:str,batch_size: int=100):
        try:
            logging.info(f"Classic upload called with index_name={index_name!r}")
            uploaded_ids=[]
            if index_name=="langchain-vector-index":
                docs,vectors=self.generate_langchain_embeddings()
            elif index_name=="langgraph-vector-index":
                docs,vectors=self.generate_langgraph_embeddings()
            index=pc.Index(index_name)
            records=[{
                "id":self._make_id(doc,i),
                "values":vec,
                "metadata":{**self._clean_metadata(doc.metadata), "text":doc.page_content}
            }
               for i,(doc,vec) in enumerate(zip(docs,vectors))
            ]
            for start in range(0,len(records),batch_size):
                batch=records[start:start+batch_size]
                index.upsert(vectors=batch, namespace="__default__")
                uploaded_ids.extend([record["id"] for record in batch])
            logging.info(f"Uploaded {len(records)} records to {index_name}")
        except Exception as e:
            logging.error(f"Error occurred while uploading to vector database: {str(e)}")
            if index is not None and uploaded_ids:
                self._rollback(index=index, ids=uploaded_ids, namespace="__default__")
            raise AgenticException(e,sys)

    def upload_to_hybrid_vector_db(self,index_name:str, batch_size: int=100):
        try:
            logging.info(f"Hybrid upload called with index_name={index_name!r}")
            uploaded_ids=[]
            if index_name=="langchain-hybrid-index":
                 docs,vectors=self.generate_langchain_embeddings()
            elif index_name=="langgraph-hybrid-index":
                docs,vectors=self.generate_langgraph_embeddings()
            index=pc.Index(index_name)
            records=[
                {
                    **self._clean_metadata(doc.metadata, hybrid=True),
                    "_id":self._make_id(doc,i),
                    "title":doc.metadata.get("title",""),
                    "description":doc.metadata.get("description",""),
                    "page_content":doc.page_content,
                    "embedding":vec
                }
                for i,(doc,vec) in enumerate(zip(docs,vectors))
            ]
            for start in range(0,len(records),batch_size):
                batch=records[start:start+batch_size]
                index.documents.upsert(namespace="__default__",documents=batch)
                uploaded_ids.extend(d["_id"] for d in batch)
            logging.info(f"Uploaded {len(records)} records to hybrid index {index_name}")
        except Exception as e:
            logging.error(f"Error occurred while uploading to hybrid vector database: {str(e)}")
            if index is not None and uploaded_ids:
                self._rollback(index=index, ids=uploaded_ids, namespace="__default__")
            raise AgenticException(e,sys)

if __name__=="__main__":
    try:
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device":"cpu"},encode_kwargs={"normalize_embeddings":True})
        ingestion=DataIngestion(document_extractor=DocumentExtractor(LANGCHAIN_DATA_SOURCE,LANGGRAPH_DATA_SOURCE,text_splitter),embeddings=embeddings)
        for index in VECTOR_INDEXES:
            logging.info("Starting to upload vectors in classic indexes")
            ingestion.upload_to_classic_vector_db(index_name=index)
            logging.info("Upload done in classic indexes")
        for index in HYBRID_INDEXES:
            logging.info("Starting to upload vectors in hybrid indexes")
            ingestion.upload_to_hybrid_vector_db(index_name=index)
            logging.info("Upload done in hybrid indexes")
    except Exception as e:
        logging.error(f"Error occurred in DataIngestion: {str(e)}")
        raise AgenticException(e,sys)