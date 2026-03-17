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
    db_dir = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
    
    print("Loading existing databases from disk...")
    
    # Load from disk instead of re-indexing!
    dense = DenseRetriever()
    dense.load(db_dir)
    
    sparse = SparseRetriever()
    sparse.load(db_dir)
    
    # Initialize Hybrid Retriever
    hybrid = HybridRetriever(dense, sparse, alpha=0.5)

    # Define Ground Truth Dataset for Technical Testing
    ground_truth_data = [
        {
            "query": "What is the equation for Scaled Dot-Product Attention?",
            "expected_source": "attention_paper.pdf"
        },
        {
            "query": "Why did the authors choose to use Multi-Head Attention instead of a single attention function?",
            "expected_source": "attention_paper.pdf"
        },
        {
            "query": "What is the default port number that the PostgreSQL server listens on?",
            "expected_source": "postgres_docs.pdf"
        },
        {
            "query": "What is the primary purpose of the Write-Ahead Log (WAL) in PostgreSQL?",
            "expected_source": "postgres_docs.pdf"
        }
    ]

    # Run Evaluator
    evaluator = RAGEvaluator(retriever=hybrid, top_k=3)
    evaluator.evaluate(ground_truth_data)