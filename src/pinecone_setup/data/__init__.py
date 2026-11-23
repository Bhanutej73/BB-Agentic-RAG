LANGGRAPH_DATA_SOURCE=[
"https://docs.langchain.com/oss/python/langgraph/overview.md",
"https://docs.langchain.com/oss/python/langgraph/workflows-agents.md",
"https://docs.langchain.com/oss/python/langgraph/graph-api#map-reduce-and-the-send-api.md"
]

LANGCHAIN_DATA_SOURCE=[
    "https://docs.langchain.com/oss/python/langchain/overview.md",
    "https://docs.langchain.com/oss/python/langchain/rag.md",
    "https://docs.langchain.com/oss/python/langchain/knowledge-base.md"
]

VECTOR_INDEXES=["langchain-vector-index","langgraph-vector-index"]

HYBRID_INDEXES=["langchain-hybrid-index","langgraph-hybrid-index"]

# VECTOR_INDEXES={
#     "langchain-vector-index":
#       {"name": "langchain-vector-index", "host":"https://langchain-vector-index-9akmou5.svc.aped-4627-b74a.pinecone.io"},
#       "langgraph-vector-index":
#       {"name":"langgraph-vector-index", "host":"https://langgraph-vector-index-9akmou5.svc.aped-4627-b74a.pinecone.io"}
# }

# HYBRID_INDEXES={
#     "langchain-hybrid-index":{"name":"langchain-hybrid-index", "host":"https://langchain-hybrid-index-9akmou5.svc.aped-4627-b74a.pinecone.io"},
    
#     "langgraph-hybrid-index":
#     {"name":"langgraph-hybrid-index", "host":"https://langgraph-hybrid-index-9akmou5.svc.aped-4627-b74a.pinecone.io"},
# }

EVALUATION_DATASET_PATH="src\\test_dataset\\golden_v1_draft.csv"

LANGCHAIN_EVALUATION_DATASET_PATH="src\\test_dataset\\Langchain_golden_v1.csv"

LANGGRAPH_EVALUATION_DATASET_PATH="src\\test_dataset\\Langgraph_golden_v1.csv"