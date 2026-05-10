import sys
import os
import re
import hashlib
import numpy as np
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

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
    """Uses LangChain's recursive splitter to chunk contextually."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def semantic_chunking(text: str, embedder, percentile_threshold: int = 90) -> List[str]:
    """Splits text into sentences, embeds them, and breaks chunks at semantic shifts."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    # Filter out empty strings or very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) == 0:
        return []
    if len(sentences) == 1:
        return sentences # No need to calculate similarity for a single sentence
    embeddings = embedder.encode(sentences)
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)
        
    # Extra safety check just in case
    if not similarities:
        return [" ".join(sentences)]
        
    threshold = np.percentile(similarities, 100 - percentile_threshold)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i, sim in enumerate(similarities):
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def process_chunks(documents: List[Dict], strategy: str = "recursive", chunk_size: int = 500, overlap: int = 50, embedder=None) -> List[Dict]:
    """Applies chunking and injects a unique chunk_id for hybrid retrieval alignment."""
    chunked_docs = []
    
    for doc in documents:
        text = doc.get("text", "")
        # Safely get and copy metadata
        base_metadata = doc.get("metadata", {}).copy()
        
        if strategy == "fixed":
            chunks = fixed_chunking(text, chunk_size)
        elif strategy == "overlap":
            chunks = overlap_chunking(text, chunk_size, overlap)
        elif strategy == "recursive":
            chunks = recursive_chunking(text, chunk_size, overlap)
        elif strategy == "semantic":
            if embedder is None:
                raise ValueError("Semantic chunking requires an 'embedder' model instance.")
            chunks = semantic_chunking(text, embedder)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 10:
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_index"] = i
                
                # Safely access source and page keys
                source = chunk_meta.get('source', 'unknown')
                page = chunk_meta.get('page', 0)
                
                unique_string = f"{source}_{page}_{i}"
                chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
                chunk_meta["chunk_id"] = chunk_id
                
                chunked_docs.append({
                    "text": chunk.strip(),
                    "metadata": chunk_meta
                })
                
    return chunked_docs

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.loader import load_directory
    from ingestion.cleaner import clean_documents
    from sentence_transformers import SentenceTransformer
    
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    print("Running ingestion pipeline: Load -> Clean -> Chunk")
    
    raw_docs = load_directory(test_dir)
    cleaned_docs = clean_documents(raw_docs)
    
    print("Loading test embedding model...")
    test_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    fixed_docs = process_chunks(cleaned_docs, strategy="fixed")
    recursive_docs = process_chunks(cleaned_docs, strategy="recursive")
    semantic_docs = process_chunks(cleaned_docs, strategy="semantic", embedder=test_embedder)
    
    print("\n--- Chunking Results ---")
    print(f"Original Pages/Segments: {len(cleaned_docs)}")
    print(f"Fixed Chunk Count: {len(fixed_docs)}")
    print(f"Recursive Chunk Count: {len(recursive_docs)}")
    print(f"Semantic Chunk Count: {len(semantic_docs)}")