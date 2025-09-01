import json
from pathlib import Path
import faiss
import numpy as np
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh import scoring
from sentence_transformers import SentenceTransformer, CrossEncoder

# Paths
ROOT = Path(__file__).parent.parent
BM25_INDEX_DIR = ROOT / "index" / "bm25_index"
FAISS_INDEX_DIR = ROOT / "index" / "faiss_index"
CHUNKS_FILE = ROOT / "data" / "processed_chunks" / "chunks.json"

# Load BM25
bm25_ix = open_dir(str(BM25_INDEX_DIR))
bm25_searcher = bm25_ix.searcher(weighting=scoring.BM25F())
bm25_parser = QueryParser("content", bm25_ix.schema)

# Load FAISS
faiss_index = faiss.read_index(str(FAISS_INDEX_DIR / "index.faiss"))
with open(FAISS_INDEX_DIR / "ids.json", 'r', encoding='utf-8') as f:
    faiss_ids = json.load(f)

# Load chunks metadata
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    chunk_data = {chunk['id']: chunk for chunk in json.load(f)}

# Embedding & reranker models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve(query: str, top_k: int = 5, alpha: float = 0.7):
    """
    Retrieve top_k chunks using BM25+FAISS fusion, heuristic boosts,
    and cross-encoder reranking.
    """
    # 1) BM25 search
    q = bm25_parser.parse(query)
    bm25_results = bm25_searcher.search(q, limit=top_k)
    bm25_scores = {hit['id']: hit.score for hit in bm25_results}

    # 2) FAISS search
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    scores, idxs = faiss_index.search(q_emb, top_k)
    dense_scores = {faiss_ids[idx]: float(scores[0][i]) for i, idx in enumerate(idxs[0])}

    # 3) Combine scores
    combined = {}
    for cid, score in bm25_scores.items():
        combined[cid] = combined.get(cid, 0) + (1 - alpha) * score
    for cid, score in dense_scores.items():
        combined[cid] = combined.get(cid, 0) + alpha * score

    # 4) Heuristic filename boost
    for cid in list(combined.keys()):
        src = chunk_data[cid]['source_file'].lower()
        if any(k in src for k in ("tokenizer", "quickstart", "getting_started",
                                   "quicktour", "tutorial", "usage", "installation")):
            combined[cid] += 2.0

    # 5) Heuristic code‐block boost
        for cid in list(combined.keys()):
            if "```" in chunk_data[cid]['content']:
                combined[cid] += 1.0

    # 6) Preliminary top_N for reranking
    top_N = min(len(combined), 20)
    # prelim is a list of (cid, combined_score)
    prelim = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_N]

    # 7) Cross‐encoder rerank
    rerank_inputs = []
    for item in prelim:
        cid = item[0]                 # first element is chunk ID
        text = chunk_data[cid]['content']
        rerank_inputs.append((query, text))
    rerank_scores = reranker.predict(rerank_inputs)
    final = [(cid, rr_score) for (cid, _), rr_score in zip(prelim, rerank_scores)]

    # 8) Final top_k selection
    final = []
    for idx, item in enumerate(prelim):
        cid = item[0]
        score = rerank_scores[idx]   # aligned by index
        final.append((cid, score))

    # 9) Build results
    top_final = sorted(final, key=lambda x: x[1], reverse=True)[:top_k]

    # 10) Build results
    results = []
    for cid, score in top_final:
        chunk = chunk_data[cid]
        results.append({
            'id': cid,
            'source_file': chunk['source_file'],
            'content': chunk['content'],
            'score': score
        })
    return results
