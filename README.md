## GenAI RAG — Transformers Documentation Assistant
 
A local, CPU-friendly Retrieval-Augmented Generation (RAG) system for Hugging Face Transformers documentation.
<br> Combines BM25 sparse search, FAISS semantic search, and a cross-encoder reranker to retrieve relevant docs, then uses a quantized LLaMA model via llama.cpp to generate concise Python code examples.

## ⚙️ Setup

#### 1. Clone the repo
```
git clone https://github.com/AP-047/docu_rag.git
```

#### 2. Install dependencies
```
pip install -r requirements.txt
```

#### 3. Build or download llama.cpp
- Option_1: Build from source (CPU only):
```
cd llama.cpp
mkdir build && cd build
cmake -DLLAMA_CURL=OFF -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```
Copy the resulting llama-cli.exe into a convenient folder.

- Option_2: Download a Windows prebuilt binary from
https://github.com/ggerganov/llama.cpp/releases
(If you are using already downloaded one here in this repo then just adjust the path at "LLAMA_CPP_BIN =" in generator.py)

#### 4. Quantize a LLaMA model
```
cd llama.cpp
./main.exe --model ./weights/llama-7B/ggml-model.bin \
           --output ../models/llama_quant/llama-7B-quant.bin \
           --quantize q4_0
```

#### 5. Update src/generator.py
Set LLAMA_CPP_BIN to the path of your llama-cli.exe.

## 🚀 Running the App

In the project root:
```
streamlit run app.py
```
Open http://localhost:8501 in your browser.
<br>
Use the sidebar sliders to adjust:
- Number of contexts (top_k)
- Dense vs Sparse (α)
- Temperature
- Max tokens
<br> <img src="data\cover_images\image_1.png" alt="Detective Profile" width="700" height="auto">


> **Type a question like:**  
> *How do I load a tokenizer in Hugging Face Transformers?*

and click Generate Answer.
<br> <img src="data\cover_images\image_2.png" alt="Detective Profile" width="700" height="auto">


## ⚙️ How It Works

#### 1. Chunking
- Raw documentation is split into ~400-word chunks.
- Each chunk includes metadata for context-aware retrieval.

#### 2. Indexing
- **BM25** (Whoosh) for keyword-based sparse search.
- **FAISS** for semantic search using embeddings (`all-MiniLM-L6-v2`).

#### 3. Retrieval
- Combine BM25 and FAISS scores for hybrid relevance.
- Apply heuristic boosts for code-rich and tutorial-heavy pages.
- Use a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) to refine top results.

#### 4. Generation
- Construct an extraction-style prompt focused on Python code output.
- Invoke a quantized LLaMA model locally via `llama.cpp`.


## 📄 Attribution
- Hugging Face Transformers documentation (CC BY 4.0)
- Whoosh, FAISS, Sentence-Transformers, Cross-Encoder (MIT/Apache2)
- llama.cpp by Georgi Gerganov (MIT)
- Streamlit (Apache2)
All components are open-source. No proprietary code included.

## 🙏 Contributions & Issues
Feel free to open issues or pull requests to improve retrieval heuristics, add GPU support, or extend to other documentation collections.
