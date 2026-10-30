import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec,SchemaBuilder
from src.pinecone_setup.data import VECTOR_INDEXES,HYBRID_INDEXES

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


hybrid_schema=(
    SchemaBuilder()
    .add_string_field(name="title", full_text_search={})
    .add_string_field("description", full_text_search={})
    .add_string_field("page_content", full_text_search={})
    .add_dense_vector_field(name="embedding", dimension=384, metric="cosine")
    .build()
)

for index in VECTOR_INDEXES:
    if not pc.has_index(index):
        pc.create_index(
            name=index,
            dimension=384,       # all-MiniLM-L6-v2 output size
            metric="cosine",     # fine for dense-only with normalized vectors
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

for index in HYBRID_INDEXES:
    if not pc.has_index(index):
        pc.indexes.create(name=index, schema=hybrid_schema)

