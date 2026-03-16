import os
from typing import List, Dict, Tuple

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha: float = 0.5):
        """
        Initializes the hybrid retriever.
        alpha (float): The weight given to the dense retriever (0.0 to 1.0).
                       (1 - alpha) will be the weight for the sparse retriever.
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def _normalize_scores(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        """Applies Min-Max normalization to a dictionary of scores."""
        if not scores_dict:
            return {}
        
        values = list(scores_dict.values())
        min_val = min(values)
        max_val = max(values)
        
        normalized = {}
        for chunk_id, score in scores_dict.items():
            if max_val == min_val:
                normalized[chunk_id] = 1.0 if max_val > 0 else 0.0
            else:
                normalized[chunk_id] = (score - min_val) / (max_val - min_val)
                
        return normalized

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Performs both searches, normalizes scores, and returns a fused ranking."""
        
        # 1. Get raw results from both retrievers (fetch more to ensure good overlap)
        fetch_k = max(top_k * 2, 10)
        dense_results = self.dense_retriever.search(query, top_k=fetch_k)
        sparse_results = self.sparse_retriever.search(query, top_k=fetch_k)
        
        # 2. Extract scores and map them by the unique chunk_id
        dense_scores = {}
        sparse_scores = {}
        chunk_map = {} # To hold the actual chunk data

        for chunk, distance in dense_results:
            chunk_id = chunk["metadata"]["chunk_id"]
            chunk_map[chunk_id] = chunk
            # Invert L2 distance so higher is better
            dense_scores[chunk_id] = 1.0 / (1.0 + distance)

        for chunk, score in sparse_results:
            chunk_id = chunk["metadata"]["chunk_id"]
            chunk_map[chunk_id] = chunk
            sparse_scores[chunk_id] = score

        # 3. Normalize both score sets to a 0.0 - 1.0 scale
        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_scores)

        # 4. Calculate weighted sum for all unique chunks found
        final_scores = {}
        all_chunk_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
        
        for chunk_id in all_chunk_ids:
            d_score = norm_dense.get(chunk_id, 0.0)
            s_score = norm_sparse.get(chunk_id, 0.0)
            
            # Weighted fusion formula
            combined = (self.alpha * d_score) + ((1 - self.alpha) * s_score)
            final_scores[chunk_id] = combined

        # 5. Sort by final score descending and return Top-K
        sorted_ids = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)
        
        fused_results = []
        for chunk_id in sorted_ids[:top_k]:
            fused_results.append((chunk_map[chunk_id], final_scores[chunk_id]))
            
        return fused_results

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.loader import load_directory
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import process_chunks
    from retrieval.dense import DenseRetriever
    from retrieval.sparse import SparseRetriever

    # 1. Ingestion Pipeline
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_docs = load_directory(test_dir)
    cleaned_docs = clean_documents(raw_docs)
    chunks = process_chunks(cleaned_docs, strategy="recursive", chunk_size=500, overlap=50)

    # 2. Initialize and Index Both Retrievers
    dense_retriever = DenseRetriever()
    dense_retriever.index_documents(chunks)
    
    sparse_retriever = SparseRetriever()
    sparse_retriever.index_documents(chunks)

    # 3. Initialize Hybrid Retriever (50% Dense / 50% Sparse)
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, alpha=0.5)

    # 4. Test Hybrid Query
    test_query = "What chunking strategies are used for TF-IDF?"
    print(f"\n--- Testing Hybrid Query ---")
    print(f"Query: '{test_query}'")
    
    results = hybrid_retriever.search(test_query, top_k=3)
    
    for i, (chunk, score) in enumerate(results):
        print(f"\nResult {i+1} (Fused Score: {score:.4f})")
        print(f"Source: {chunk['metadata'].get('source', 'Unknown')} (Page {chunk['metadata'].get('page', 'N/A')})")
        print(f"Text snippet: {chunk['text'][:200]}...")