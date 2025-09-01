import os
import json
import re
from pathlib import Path
from typing import List, Dict

# Define paths
ROOT = Path(__file__).parent.parent
RAW_DOCS_DIR = ROOT / "data" / "raw_docs" / "transformers" / "docs"
PROCESSED_DIR = ROOT / "data" / "processed_chunks"

def clean_markdown_content(content: str) -> str:
    """Clean markdown content by removing front matter, excessive whitespace, etc."""
    # Remove YAML front matter
    content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    
    # Clean up multiple newlines
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Remove excessive whitespace
    content = re.sub(r'[ \t]+', ' ', content)
    
    return content.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks based on word count."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

def process_markdown_file(file_path: Path) -> List[Dict]:
    """Process a single markdown file and return chunks with metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Clean the content
        clean_content = clean_markdown_content(content)
        
        # Skip empty files
        if len(clean_content.strip()) < 50:
            return []
        
        # Create chunks
        chunks = chunk_text(clean_content)
        
        # Create chunk records with metadata
        chunk_records = []
        relative_path = file_path.relative_to(RAW_DOCS_DIR)
        
        for i, chunk in enumerate(chunks):
            chunk_record = {
                'id': f"{relative_path}_{i}",
                'source_file': str(relative_path),
                'chunk_index': i,
                'content': chunk,
                'word_count': len(chunk.split())
            }
            chunk_records.append(chunk_record)
        
        return chunk_records
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_all_docs():
    """Process all markdown files in the docs directory."""
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown files
    md_files = list(RAW_DOCS_DIR.glob("**/*.md"))
    print(f"Found {len(md_files)} markdown files")
    
    all_chunks = []
    processed_files = 0
    
    for md_file in md_files:
        print(f"Processing: {md_file.relative_to(RAW_DOCS_DIR)}")
        chunks = process_markdown_file(md_file)
        all_chunks.extend(chunks)
        processed_files += 1
        
        if processed_files % 10 == 0:
            print(f"Processed {processed_files}/{len(md_files)} files...")
    
    # Save all chunks to JSON file
    output_file = PROCESSED_DIR / "chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete!")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Output saved to: {output_file}")
    
    # Print some statistics
    word_counts = [chunk['word_count'] for chunk in all_chunks]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    
    print(f"Average chunk size: {avg_words:.1f} words")
    print(f"Processed files: {processed_files}")

if __name__ == "__main__":
    process_all_docs()