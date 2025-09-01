import os
import json
import faiss
from pathlib import Path
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh import scoring
from sentence_transformers import SentenceTransformer
import numpy as np

# Paths
ROOT = Path(__file__).parent.parent
BM25_INDEX_DIR = ROOT / "index" / "bm25_index"
FAISS_INDEX_DIR = ROOT / "index" / "faiss_index"
CHUNKS_FILE = ROOT / "data" / "processed_chunks" / "chunks.json"

# Load FAISS index and IDs
faiss_index = faiss.read_index(str(FAISS_INDEX_DIR / "index.faiss"))
with open(FAISS_INDEX_DIR / "ids.json", 'r', encoding='utf-8') as f:
    faiss_ids = json.load(f)

# Load chunk metadata
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    chunk_data = {chunk['id']: chunk for chunk in json.load(f)}

# Load BM25 index
bm25_ix = open_dir(str(BM25_INDEX_DIR))
bm25_searcher = bm25_ix.searcher(weighting=scoring.BM25F())
bm25_parser = QueryParser("content", bm25_ix.schema)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, top_k: int = 5, alpha: float = 0.5):
    """
    Retrieve top_k chunks combining BM25 and FAISS.
    alpha: weight for FAISS (dense), (1-alpha) for BM25 (sparse).
    """
    # BM25 search
    q = bm25_parser.parse(query)
    bm25_results = bm25_searcher.search(q, limit=top_k)
    bm25_scores = {hit['id']: hit.score for hit in bm25_results}

    # FAISS search
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    scores, idxs = faiss_index.search(q_emb, top_k)
    dense_scores = {faiss_ids[idx]: float(scores[0][i]) for i, idx in enumerate(idxs[0])}

    # Combine scores
    combined = {}
    for cid, score in bm25_scores.items():
        combined[cid] = combined.get(cid, 0) + (1 - alpha) * score
    for cid, score in dense_scores.items():
        combined[cid] = combined.get(cid, 0) + alpha * score

    # Sort by combined score
    sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Gather results
    results = []
    for cid, score in sorted_ids:
        chunk = chunk_data[cid]
        results.append({
            'id': cid,
            'source_file': chunk['source_file'],
            'content': chunk['content'],
            'score': score
        })
    return results

# For testing
if __name__ == "__main__":
    for query in ["transformer attention", "tokenizer hugging face"]:
        print(f"\nQuery: {query}")
        for r in retrieve(query, top_k=3, alpha=0.5):
            print(f"\n- ID: {r['id']}, Score: {r['score']:.4f}")
            print(f"  Source: {r['source_file']}")
            print(f"  Preview: {r['content'][:150]}...")