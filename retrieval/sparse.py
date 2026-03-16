import os
import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import pickle

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer that lowercases text and extracts alphanumeric words.
    Essential for BM25 term frequency calculations.
    """
    return re.findall(r'\w+', text.lower())

class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def index_documents(self, chunks: List[Dict]):
        """Tokenizes text chunks and builds the BM25 index."""
        if not chunks:
            print("No chunks provided for indexing.")
            return

        self.chunks = chunks
        print(f"Tokenizing {len(chunks)} chunks for sparse retrieval...")
        
        # Tokenize the text for each chunk
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index successfully built for {len(tokenized_corpus)} documents.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Searches the BM25 index for exact keyword matches."""
        if not self.bm25:
            print("BM25 Index is empty. Please index documents first.")
            return []

        # Tokenize the query
        tokenized_query = tokenize(query)
        
        # Get raw BM25 scores for all chunks
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score in descending order (higher score = better match)
        # We use standard Python sorting here since it's an in-memory array
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = doc_scores[idx]
            # Only return results that actually matched at least one term (score > 0)
            if score > 0:
                results.append((self.chunks[idx], float(score)))
                
        return results
    
    def save(self, save_dir: str):
        """Saves the BM25 model and chunks to disk."""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "sparse.pkl"), "wb") as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)

    def load(self, save_dir: str):
        """Loads the BM25 model and chunks from disk."""
        with open(os.path.join(save_dir, "sparse.pkl"), "rb") as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']

if __name__ == "__main__":
    # Test keyword query integration
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.loader import load_directory
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import process_chunks

    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # 1. Run ingestion pipeline
    raw_docs = load_directory(test_dir)
    cleaned_docs = clean_documents(raw_docs)
    chunks = process_chunks(cleaned_docs, strategy="recursive", chunk_size=500, overlap=50)

    # 2. Initialize and index sparse retriever
    retriever = SparseRetriever()
    retriever.index_documents(chunks)

    # 3. Test exact keyword query
    # Using specific terms from your Major Project Synopsis document
    test_query = "BM25 or TF-IDF"
    print(f"\n--- Testing Sparse/Keyword Query ---")
    print(f"Query: '{test_query}'")
    
    results = retriever.search(test_query, top_k=3)
    
    if not results:
        print("No matches found.")
    else:
        for i, (chunk, score) in enumerate(results):
            print(f"\nResult {i+1} (BM25 Score: {score:.4f})")
            print(f"Source: {chunk['metadata'].get('source', 'Unknown')}")
            print(f"Text snippet: {chunk['text'][:200]}...")