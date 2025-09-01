# GenAI RAG — Transformers Documentation Assistant
 ---
 
A local, CPU-friendly Retrieval-Augmented Generation (RAG) system for Hugging Face Transformers documentation.
<br> Combines BM25 sparse search, FAISS semantic search, and a cross-encoder reranker to retrieve relevant docs, then uses a quantized LLaMA model via llama.cpp to generate concise Python code examples.

# ⚙️ Setup
 ---

1. Clone the repo
```
git clone https://github.com/your-username/docu_rag.git
cd docu_rag
```

2. Create and activate a Python virtual environment
```
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Build or download llama.cpp
Build from source (CPU only):
```
cd llama.cpp
mkdir build && cd build
cmake -DLLAMA_CURL=OFF -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```
Copy the resulting llama-cli.exe into a convenient folder.

- Or download a Windows prebuilt binary from
https://github.com/ggerganov/llama.cpp/releases

5. Quantize a LLaMA model
```
cd llama.cpp
./main.exe --model ./weights/llama-7B/ggml-model.bin \
           --output ../models/llama_quant/llama-7B-quant.bin \
           --quantize q4_0
```

6. Update src/generator.py
Set LLAMA_CPP_BIN to the path of your llama-cli.exe.

🚀 Running the App
---

In the project root:
streamlit run app.py
Open http://localhost:8501 in your browser.
Use the sidebar sliders to adjust:

Number of contexts (top_k)

Dense vs Sparse (α)

Temperature

Max tokens

Type a question like:

How do I load a tokenizer in Hugging Face Transformers?

and click Generate Answer.
