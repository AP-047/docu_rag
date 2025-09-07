## GenAI RAG — Transformers Documentation Assistant
 
A **local**, **CPU-friendly** **semantic search** system for Hugging Face Transformers documentation.
<br> Combines BM25 sparse search, FAISS embeddings, and a cross-encoder reranker to retrieve and display the most relevant documentation—**fully offline** and **without external dependencies**.

✅ Fully offline • ✅ 0% Hallucination • ✅ CPU-friendly • ✅ Hybrid retrieval • ✅ Real-time search

## ⚙️ Setup

### 1. Clone the repo
```
git clone https://github.com/AP-047/docu_rag.git
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

## 🚀 Running the App

From the project root, launch Streamlit:
```
streamlit run app.py
```
Open http://localhost:8501 in your browser.
<br>
Use the sidebar sliders to adjust:
- Number of contexts (top_k)
- Dense vs Sparse balance (α)
<br> <img src="data\cover_images\image_1.png" alt="Detective Profile" width="700" height="auto">

Enter a query and click **Search Documentation.**  
<br> *How do I perform text classification using Transformers?*
<br> *How do I load a tokenizer in Transformers?*
<br> *How do I load a tokenizer in Hugging Face Transformers?*

and click Generate Answer.
<br> <img src="data\cover_images\image_2.png" alt="Detective Profile" width="700" height="auto">


## ⚙️ How It Works

### 1. Chunking
- Splits each Markdown file into ~400-word passages with associated metadata.

### 2. Indexing
- **BM25** (Whoosh) for keyword-based sparse search.
- **FAISS** for semantic search using all-MiniLM-L6-v2 embeddings.

### 3. Retrieval & Reranking
- Combine BM25 and FAISS scores via a hybrid α-weighted sum.
- Applies a cross-encoder (ms-marco-MiniLM-L-6-v2) to rerank the top results.

### 4. Display
- Shows each passage with its source path and relevance score.

## 🌐 Live Demo

**Try it now:** [https://transformers-doc-assistant.streamlit.app/](https://transformers-doc-assistant.streamlit.app)
<br> No installation required! Test the semantic search capabilities directly in your browser.

## 📄 Attribution
- Hugging Face Transformers documentation (CC BY 4.0)
- Whoosh, FAISS, Sentence-Transformers, Cross-Encoder (MIT/Apache2)
- Streamlit (Apache2)
All components are open-source. No proprietary code included.

## 🙏 Contributions & Issues
Feel free to open issues or pull requests to improve retrieval heuristics, add GPU support, or extend to other documentation collections.
