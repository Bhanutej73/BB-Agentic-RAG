| Column | Source | Meaning |
|---|---|---|
| `user_input` | Test set generator | The synthetic question |
| `reference_contexts` | Test set generator | Ideal chunks that should be retrieved (ground truth) |
| `reference` | Test set generator | Ideal answer written from those chunks (ground truth) |
| `retrieved_contexts` | **Your retriever** | Chunks your RAG system actually retrieved |
| `response` | **Your LLM** | Answer your RAG system actually generated |