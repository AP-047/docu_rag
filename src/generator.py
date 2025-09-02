import os
import subprocess
from pathlib import Path
from src.retriever import retrieve

# Paths
ROOT = Path(__file__).parent.parent
LLAMA_MODEL_PATH = ROOT / "models" / "llama_quant" / "llama-7B-quant.bin"
LLAMA_CPP_BIN = r"C:/AP_nxt/projects/docu_rag/llama-b6347-bin-win-cpu-x64/llama-cli.exe"
# LLAMA_CPP_BIN = r"C:\AP_nxt\projects\docu_rag\llama-b6347-bin-win-cpu-x64\llama-cli.exe"

# Defaults (overridden by UI)
TEMPERATURE = 0.5
TOP_P = 0.9
MAX_TOKENS = 512

def build_prompt(query: str, contexts: list) -> str:
    """
    Construct an extraction‐style prompt that asks for a concise
    Python code snippet, citing context IDs.
    """
    header = (
        "You are a code assistant. Given the following documentation contexts, "
        "extract and return *only* the **Python code snippet** that answers the question. "
        "If no code snippet exists in context, respond “No example found.” "
        "Cite each code block with its context ID.\n\n"
    )
    context_text = ""
    for ctx in contexts:
        # Escape backticks and preserve fences
        content = ctx["content"].replace("```", "\\`\\`\\`")
        context_text += f"[{ctx['id']}] ``````\n\n"
    return header + context_text + f"Question: {query}\n\nAnswer (code only):"

def generate_answer(query: str, top_k: int = 5, alpha: float = 0.7) -> str:
    """
    Retrieve contexts, build the extraction prompt, and invoke llama-cli
    to generate the code answer.
    """
    contexts = retrieve(query, top_k=top_k, alpha=alpha)
    prompt = build_prompt(query, contexts)

    cmd = [
        LLAMA_CPP_BIN,
        "--model", str(LLAMA_MODEL_PATH),
        "--prompt", prompt,
        "--n_predict", str(MAX_TOKENS),
        "--temp", str(TEMPERATURE),
        "--top_p", str(TOP_P),
        "--threads", str(os.cpu_count()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout.strip()