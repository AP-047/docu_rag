import os
import subprocess

# Define target paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw_docs", "transformers")

def clone_transformers_docs():
    if os.path.exists(RAW_DIR):
        print(f"Directory already exists: {RAW_DIR}")
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    repo_url = "https://github.com/huggingface/transformers.git"
    subprocess.run(["git", "clone", "--depth", "1", repo_url, RAW_DIR], check=True)
    print("Cloned Transformers repository.")

if __name__ == "__main__":
    clone_transformers_docs()