import sys
import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

def fixed_chunking(text: str, chunk_size: int = 500) -> List[str]:
    """Splits text into fixed-size chunks without overlap."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def overlap_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits text into chunks with a sliding window overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap")
    
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks

def recursive_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Uses LangChain's recursive splitter to chunk contextually (paragraphs, sentences, words)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def process_chunks(documents: List[Dict], strategy: str = "recursive", chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Applies chunking and injects a unique chunk_id for hybrid retrieval alignment."""
    chunked_docs = []
    
    for doc in documents:
        text = doc["text"]
        base_metadata = doc["metadata"]
        
        if strategy == "fixed":
            chunks = fixed_chunking(text, chunk_size)
        elif strategy == "overlap":
            chunks = overlap_chunking(text, chunk_size, overlap)
        elif strategy == "recursive":
            chunks = recursive_chunking(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 10:
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_index"] = i
                
                # Generate a unique deterministic ID for hybrid score fusion mapping
                unique_string = f"{chunk_meta['source']}_{chunk_meta.get('page', 0)}_{i}"
                chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
                chunk_meta["chunk_id"] = chunk_id
                
                chunked_docs.append({
                    "text": chunk.strip(),
                    "metadata": chunk_meta
                })
                
    return chunked_docs

if __name__ == "__main__":
    # Test chunk count and pipeline integration
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.loader import load_directory
    from ingestion.cleaner import clean_documents
    
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    print("Running ingestion pipeline: Load -> Clean -> Chunk")
    
    raw_docs = load_directory(test_dir)
    cleaned_docs = clean_documents(raw_docs)
    
    # Test all three strategies to compare chunk counts
    fixed_docs = process_chunks(cleaned_docs, strategy="fixed")
    overlap_docs = process_chunks(cleaned_docs, strategy="overlap")
    recursive_docs = process_chunks(cleaned_docs, strategy="recursive")
    
    print("\n--- Chunking Results ---")
    print(f"Original Cleaned Pages/Segments: {len(cleaned_docs)}")
    print(f"Fixed Chunk Count: {len(fixed_docs)}")
    print(f"Overlap Chunk Count: {len(overlap_docs)}")
    print(f"Recursive Chunk Count: {len(recursive_docs)}")
    
    if recursive_docs:
        print("\n--- Sample Recursive Chunk (First 200 chars) ---")
        print(recursive_docs[0]["text"][:200])
        print(f"\nMetadata: {recursive_docs[0]['metadata']}")