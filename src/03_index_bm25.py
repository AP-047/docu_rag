import os
import json
from pathlib import Path
from whoosh import fields
from whoosh.index import create_index, exists_in
from whoosh.writing import IndexWriter
from whoosh.qparser import QueryParser
from whoosh import scoring
import time

# Define paths
ROOT = Path(__file__).parent.parent
CHUNKS_FILE = ROOT / "data" / "processed_chunks" / "chunks.json"
BM25_INDEX_DIR = ROOT / "index" / "bm25_index"

def create_schema():
    """Create the search schema for our documents."""
    schema = fields.Schema(
        id=fields.ID(stored=True, unique=True),
        content=fields.TEXT(stored=True, analyzer=fields.StandardAnalyzer()),
        source_file=fields.TEXT(stored=True),
        chunk_index=fields.NUMERIC(stored=True),
        word_count=fields.NUMERIC(stored=True)
    )
    return schema

def load_chunks():
    """Load processed chunks from JSON file."""
    print(f"Loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def build_bm25_index():
    """Build BM25 index using Whoosh."""
    # Create index directory
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    chunks = load_chunks()
    
    # Create schema and index
    schema = create_schema()
    
    if exists_in(str(BM25_INDEX_DIR)):
        print("Index already exists. Removing old index...")
        import shutil
        shutil.rmtree(BM25_INDEX_DIR)
        BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Creating new BM25 index...")
    ix = create_index(str(BM25_INDEX_DIR), schema)
    
    # Add documents to index
    start_time = time.time()
    writer = ix.writer()
    
    for i, chunk in enumerate(chunks):
        try:
            writer.add_document(
                id=chunk['id'],
                content=chunk['content'],
                source_file=chunk['source_file'],
                chunk_index=chunk['chunk_index'],
                word_count=chunk['word_count']
            )
            
            # Progress update every 500 chunks
            if (i + 1) % 500 == 0:
                print(f"Indexed {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            print(f"Error indexing chunk {chunk['id']}: {e}")
            continue
    
    # Commit the index
    print("Committing index...")
    writer.commit()
    
    elapsed_time = time.time() - start_time
    print(f"\nBM25 indexing complete!")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Indexed {len(chunks)} chunks")
    print(f"Index saved to: {BM25_INDEX_DIR}")
    
    return ix

def test_bm25_search(ix, query_text="transformer model", top_k=5):
    """Test the BM25 index with a sample query."""
    print(f"\n--- Testing BM25 Search ---")
    print(f"Query: '{query_text}'")
    
    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        parser = QueryParser("content", ix.schema)
        query = parser.parse(query_text)
        
        start_time = time.time()
        results = searcher.search(query, limit=top_k)
        search_time = time.time() - start_time
        
        print(f"Search completed in {search_time:.4f} seconds")
        print(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"   Source: {result['source_file']}")
            print(f"   ID: {result['id']}")
            print(f"   Content preview: {result['content'][:150]}...")

if __name__ == "__main__":
    # Build the index
    ix = build_bm25_index()
    
    # Test with a sample query
    test_bm25_search(ix, "transformer model attention", top_k=3)
    test_bm25_search(ix, "hugging face tokenizer", top_k=3)