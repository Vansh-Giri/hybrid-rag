import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle

class DenseRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initializes the embedding model and FAISS index."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # L2 distance index (Euclidean). Lower distance = higher similarity.
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []

    def index_documents(self, chunks: List[Dict]):
        """Embeds text chunks and adds them to the FAISS index."""
        if not chunks:
            print("No chunks provided for indexing.")
            return

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Creating dense embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        print("Adding embeddings to FAISS index...")
        self.index.add(embeddings)
        print(f"FAISS index now contains {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Searches for the most semantically similar chunks to the query."""
        if self.index.ntotal == 0:
            print("Index is empty. Please index documents first.")
            return []

        # Embed the query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        # Return matched chunks and their L2 distance scores
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(distances[0][i])))
                
        return results
    
    def save(self, save_dir: str):
        """Saves the FAISS index and chunks to disk."""
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "dense.index"))
        with open(os.path.join(save_dir, "dense_chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, save_dir: str):
        """Loads the FAISS index and chunks from disk."""
        self.index = faiss.read_index(os.path.join(save_dir, "dense.index"))
        with open(os.path.join(save_dir, "dense_chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

if __name__ == "__main__":
    # Test semantic query integration
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.loader import load_directory
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import process_chunks

    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # 1. Run ingestion pipeline
    raw_docs = load_directory(test_dir)
    cleaned_docs = clean_documents(raw_docs)
    # Using recursive chunking for best semantic integrity
    chunks = process_chunks(cleaned_docs, strategy="recursive", chunk_size=500, overlap=50)

    # 2. Initialize and index
    retriever = DenseRetriever()
    retriever.index_documents(chunks)

    # 3. Test semantic query
    test_query = "What is the main architecture or framework discussed?"
    print(f"\n--- Testing Semantic Query ---")
    print(f"Query: '{test_query}'")
    
    results = retriever.search(test_query, top_k=3) # [cite: 43, 44]
    
    for i, (chunk, score) in enumerate(results):
        print(f"\nResult {i+1} (L2 Distance: {score:.4f})")
        print(f"Source: {chunk['metadata'].get('source', 'Unknown')}")
        print(f"Text snippet: {chunk['text'][:200]}...")