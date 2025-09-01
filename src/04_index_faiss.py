import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# Define paths
ROOT = Path(__file__).parent.parent
CHUNKS_FILE = ROOT / "data" / "processed_chunks" / "chunks.json"
FAISS_INDEX_DIR = ROOT / "index" / "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # embedding dimension for all-MiniLM-L6-v2

def load_chunks():
    """Load processed chunks from JSON file."""
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_index_dir():
    """Create or clear the FAISS index directory."""
    if FAISS_INDEX_DIR.exists():
        for f in FAISS_INDEX_DIR.iterdir():
            f.unlink()
    else:
        FAISS_INDEX_DIR.mkdir(parents=True)

def build_faiss_index():
    """Build a FAISS IndexFlatIP for dense retrieval."""
    # Load chunks
    chunks = load_chunks()
    texts = [chunk['content'] for chunk in chunks]
    ids = [chunk['id'] for chunk in chunks]

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Compute embeddings in batches
    batch_size = 64
    embeddings = []
    start_time = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_emb)
        print(f"Encoded {i + len(batch_texts)}/{len(texts)} chunks")
    embeddings = np.vstack(embeddings).astype('float32')

    # Create FAISS index (Inner Product for cosine similarity)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    elapsed = time.time() - start_time
    print(f"\nComputed embeddings and built index in {elapsed:.2f} seconds")
    print(f"Total vectors indexed: {index.ntotal}")

    # Save index and metadata
    ensure_index_dir()
    faiss.write_index(index, str(FAISS_INDEX_DIR / "index.faiss"))
    with open(FAISS_INDEX_DIR / "ids.json", 'w', encoding='utf-8') as f:
        json.dump(ids, f, indent=2, ensure_ascii=False)
    print(f"FAISS index saved to: {FAISS_INDEX_DIR}")

    return index, ids

def test_faiss_search(index, ids, query_text="transformer model", top_k=5):
    """Test FAISS index by encoding a query and retrieving nearest chunks."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    q_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    
    start = time.time()
    scores, idxs = index.search(q_emb, top_k)
    elapsed = time.time() - start

    print(f"\n--- Testing FAISS Search ---")
    print(f"Query: '{query_text}'")
    print(f"Search completed in {elapsed:.4f} seconds")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        chunk_id = ids[idx]
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   ID: {chunk_id}")

if __name__ == "__main__":
    index, ids = build_faiss_index()
    test_faiss_search(index, ids, "transformer attention", top_k=3)
    test_faiss_search(index, ids, "tokenizer hugging face", top_k=3)