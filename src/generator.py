import os
import subprocess
from pathlib import Path
from src.retriever import retrieve

# Project root
ROOT = Path(__file__).parent.parent

# Path to your quantized LLaMA model
LLAMA_MODEL_PATH = ROOT / "models" / "llama_quant" / "llama-7B-quant.bin"

# Path to the llama.cpp CLI executable you downloaded
LLAMA_CPP_BIN = r"C:\AP_nxt\projects\llama-b6347-bin-win-cpu-x64\llama-cli.exe"

# Generation hyperparameters (can be overridden by UI)
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 256

def build_prompt(query: str, contexts: list) -> str:
    """
    Construct the prompt by concatenating instruction header,
    retrieved contexts (with IDs), and the user question.
    """
    header = (
        "You are an expert assistant for Hugging Face Transformers documentation.\n"
        "Use the following context to answer accurately. Cite by chunk ID.\n\n"
    )
    context_text = "".join(f"[{c['id']}] {c['content']}\n\n" for c in contexts)
    return header + context_text + f"Question: {query}\nAnswer:"

def generate_answer(query: str, top_k: int = 5, alpha: float = 0.5) -> str:
    """
    Retrieve relevant chunks, build the prompt, and invoke llama-cli
    to generate a grounded answer.
    """
    # Retrieve top-k contexts
    contexts = retrieve(query, top_k=top_k, alpha=alpha)

    # Build the combined prompt
    prompt = build_prompt(query, contexts)

    # llama-cli command arguments
    cmd = [
        LLAMA_CPP_BIN,
        "--model", str(LLAMA_MODEL_PATH),
        "--prompt", prompt,
        "--n_predict", str(MAX_TOKENS),
        "--temp", str(TEMPERATURE),
        "--top_p", str(TOP_P),
        "--threads", str(os.cpu_count()),
    ]

    # Run llama-cli and capture its output
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout.strip()