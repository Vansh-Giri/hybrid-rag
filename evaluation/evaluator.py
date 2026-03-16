import time
import tracemalloc
import os
import sys
from typing import List, Dict

# Ensure we can import from our local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import load_directory
from ingestion.cleaner import clean_documents
from ingestion.chunker import process_chunks
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever

class RAGEvaluator:
    def __init__(self, retriever, top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k

    def evaluate(self, ground_truth: List[Dict]):
        """Runs the evaluation pipeline for latency, memory, precision, and recall."""
        print(f"--- Starting Evaluation (Top-{self.top_k}) ---")
        
        total_latency = 0
        hits = 0  # For Recall calculation
        total_relevant_retrieved = 0 # For Precision calculation
        
        # Start tracking memory allocation
        tracemalloc.start()

        for item in ground_truth:
            query = item["query"]
            expected_source = item["expected_source"]
            
            # Measure Latency
            start_time = time.time()
            results = self.retriever.search(query, top_k=self.top_k)
            latency = time.time() - start_time
            total_latency += latency
            
            # Check for matches
            retrieved_sources = [res[0]["metadata"].get("source", "") for res in results]
            
            # Did the expected source appear in the results? (Recall)
            is_hit = any(expected_source in source for source in retrieved_sources)
            if is_hit:
                hits += 1
                
            # How many of the retrieved results were from the expected source? (Precision)
            relevant_count = sum(1 for source in retrieved_sources if expected_source in source)
            total_relevant_retrieved += relevant_count

        # Stop memory tracking
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate final metrics
        avg_latency = total_latency / len(ground_truth)
        recall = hits / len(ground_truth)
        precision = total_relevant_retrieved / (len(ground_truth) * self.top_k)
        
        print("\n--- Evaluation Metrics ---")
        print(f"Average Retrieval Latency: {avg_latency:.4f} seconds")
        print(f"Peak Memory Usage: {peak_mem / 10**6:.2f} MB")
        print(f"Recall@{self.top_k}: {recall:.2%} (Found the target doc in {hits}/{len(ground_truth)} queries)")
        print(f"Precision@{self.top_k}: {precision:.2%} (Relevant docs / Total retrieved docs)")
        
        return {
            "latency": avg_latency,
            "memory_mb": peak_mem / 10**6,
            "recall": recall,
            "precision": precision
        }

if __name__ == "__main__":
    # 1. Boot up the Retrieval Pipeline
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_docs = load_directory(test_dir)
    cleaned_docs = clean_documents(raw_docs)
    chunks = process_chunks(cleaned_docs, strategy="recursive", chunk_size=500, overlap=50)

    dense = DenseRetriever()
    dense.index_documents(chunks)
    sparse = SparseRetriever()
    sparse.index_documents(chunks)
    
    # Test with 50/50 fusion weight
    hybrid = HybridRetriever(dense, sparse, alpha=0.5)

    # 2. Define Ground Truth Dataset
    # We use excerpts from your synopsis PDF as the expected source
    ground_truth_data = [
        {
            "query": "What are the hardware requirements?",
            "expected_source": "Major Project Synopsis.pdf"
        },
        {
            "query": "What chunking strategies are used?",
            "expected_source": "Major Project Synopsis.pdf"
        },
        {
            "query": "BM25 or TF-IDF",
            "expected_source": "Major Project Synopsis.pdf"
        }
    ]

    # 3. Run Evaluator
    evaluator = RAGEvaluator(retriever=hybrid, top_k=3)
    evaluator.evaluate(ground_truth_data)