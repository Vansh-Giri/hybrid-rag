import os
import sys

# Ensure imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY
from ingestion.loader import load_directory
from ingestion.cleaner import clean_documents
from ingestion.chunker import process_chunks
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever

def build_or_load_retrievers():
    """Conditionally loads existing indices or builds them from scratch."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    dense_path = os.path.join(INDEX_DIR, "dense.index")
    sparse_path = os.path.join(INDEX_DIR, "sparse.pkl")
    
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()

    # Conditional Execution: Check if both index files exist
    if os.path.exists(dense_path) and os.path.exists(sparse_path):
        print("Indices found on disk. Loading into memory...")
        dense_retriever.load(INDEX_DIR)
        sparse_retriever.load(INDEX_DIR)
        print("Indices loaded successfully.")
    else:
        print("Indices not found. Building from scratch...")
        # 1. Run ingestion pipeline
        raw_docs = load_directory(DATA_DIR)
        if not raw_docs:
            raise FileNotFoundError(f"No documents found in {DATA_DIR}. Please add PDFs or TXT files.")
            
        cleaned_docs = clean_documents(raw_docs)
        chunks = process_chunks(
            cleaned_docs, 
            strategy=CHUNKING_STRATEGY, 
            chunk_size=CHUNK_SIZE, 
            overlap=CHUNK_OVERLAP
        )
        
        # 2. Build indices
        dense_retriever.index_documents(chunks)
        sparse_retriever.index_documents(chunks)
        
        # 3. Save to persistent storage
        print("💾 Saving indices to disk...")
        dense_retriever.save(INDEX_DIR)
        sparse_retriever.save(INDEX_DIR)
        print("Indices built and saved successfully.")

    return dense_retriever, sparse_retriever

def get_hybrid_retriever(alpha: float = 0.5):
    """Wrapper to instantiate the Hybrid Retriever using managed indices."""
    dense, sparse = build_or_load_retrievers()
    return HybridRetriever(dense, sparse, alpha=alpha)

if __name__ == "__main__":
    # Test the manager
    from config.config import HYBRID_ALPHA
    
    print("--- Initializing Retrieval Manager ---")
    retriever = get_hybrid_retriever(alpha=HYBRID_ALPHA)
    
    print("\n--- Testing Persistence (Run this twice to see load speed) ---")
    results = retriever.search("What is RAG?", top_k=2)
    for i, (chunk, score) in enumerate(results):
         print(f"Result {i+1} Score: {score:.4f} | Chunk ID: {chunk['metadata']['chunk_id']}")