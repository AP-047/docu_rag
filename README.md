# GenAI RAG — Transformers Documentation Assistant

A local, CPU-friendly Retrieval-Augmented Generation (RAG) system for Hugging Face Transformers documentation.
<br> Combines BM25 sparse search, FAISS semantic search, and a cross-encoder reranker to retrieve relevant docs, then uses a quantized LLaMA model via llama.cpp to generate concise Python code examples.
