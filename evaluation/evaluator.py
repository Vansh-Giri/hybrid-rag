import time
import tracemalloc
import os
import sys
import json
import requests
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from our local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever

API_URL = "http://127.0.0.1:8000/query"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")

class RAGEvaluator:
    def __init__(self, retriever, top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k

    def evaluate(self, ground_truth: List[Dict]):
        """Runs the evaluation pipeline for latency, memory, precision, and recall."""
        total_latency = 0
        hits = 0
        total_relevant_retrieved = 0
        
        tracemalloc.start()

        for item in ground_truth:
            query = item["query"]
            expected_source = item["expected_source"]
            
            start_time = time.time()
            results = self.retriever.search(query, top_k=self.top_k)
            latency = time.time() - start_time
            total_latency += latency
            
            retrieved_sources = [res[0]["metadata"].get("source", "") for res in results]
            
            is_hit = any(expected_source in source for source in retrieved_sources)
            if is_hit:
                hits += 1
                
            relevant_count = sum(1 for source in retrieved_sources if expected_source in source)
            total_relevant_retrieved += relevant_count

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        avg_latency = total_latency / len(ground_truth)
        recall = hits / len(ground_truth)
        precision = total_relevant_retrieved / (len(ground_truth) * self.top_k)
        
        return {
            "latency": avg_latency,
            "memory_mb": peak_mem / 10**6,
            "recall": recall,
            "precision": precision
        }

def evaluate_generation(ground_truth: List[Dict], provider="gemini", alpha=0.5):
    """Hits the FastAPI backend and uses Groq as an LLM-Judge to score the generated answers."""
    print(f"\n--- Starting LLM-as-a-Judge Generation Eval ({provider.upper()}) ---")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Skipping Generation Eval: GROQ_API_KEY not found in .env")
        return

    client = Groq(api_key=groq_api_key)
    results = []

    for item in ground_truth:
        query = item["query"]
        print(f"Evaluating Query: {query}")
        
        payload = {"query": query, "top_k": 3, "alpha": alpha, "provider": provider}
        
        try:
            # 1. Query the live RAG API
            res = requests.post(API_URL, json=payload)
            if res.status_code != 200:
                print(f"API Error: {res.text}")
                continue
                
            data = res.json()
            answer = data["answer"]
            latency = data["latency_seconds"]
            
            # 2. Use Groq to grade the answer
            eval_prompt = f"""
            Evaluate the following RAG system answer based on the user's question.
            Question: {query}
            Answer: {answer}
            
            Score the answer from 1 to 10 on two metrics:
            1. accuracy (Is it factually correct and hallucination-free?)
            2. completeness (Does it fully answer the prompt without missing details?)
            
            Output ONLY a valid JSON object in this exact format:
            {{"accuracy": 8, "completeness": 9}}
            """
            
            eval_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": eval_prompt}],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            scores = json.loads(eval_completion.choices[0].message.content)
            
            results.append({
                "query": query,
                "answer": answer,
                "latency": latency,
                "accuracy": scores.get("accuracy", 0),
                "completeness": scores.get("completeness", 0),
                "alpha": alpha,
                "provider": provider
            })
            
        except Exception as e:
            print(f"Error evaluating generation for '{query}': {e}")
            
    # Save results to JSON for the Streamlit Dashboard
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Generation Evaluation complete. Saved to {RESULTS_FILE}")

if __name__ == "__main__":
    db_dir = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
    
    print("Loading existing databases from disk...")
    dense = DenseRetriever()
    dense.load(db_dir)
    
    sparse = SparseRetriever()
    sparse.load(db_dir)
    
    hybrid = HybridRetriever(dense, sparse, alpha=0.5)

    # Ground Truth Dataset
    ground_truth_data = [
        {"query": "What is the equation for Scaled Dot-Product Attention?", "expected_source": "attention_paper.pdf"},
        {"query": "Why did the authors choose to use Multi-Head Attention instead of a single attention function?", "expected_source": "attention_paper.pdf"},
        {"query": "What is the default port number that the PostgreSQL server listens on?", "expected_source": "postgres_docs.pdf"},
        {"query": "What is the primary purpose of the Write-Ahead Log (WAL) in PostgreSQL?", "expected_source": "postgres_docs.pdf"}
    ]

    # --- Phase 1: Retrieval Evaluation ---
    models_to_test = [
        ("Sparse (BM25)", sparse),
        ("Dense (FAISS)", dense),
        ("Hybrid (Alpha=0.5)", hybrid)
    ]

    results_table = []

    for name, retriever_instance in models_to_test:
        evaluator = RAGEvaluator(retriever=retriever_instance, top_k=3)
        metrics = evaluator.evaluate(ground_truth_data)
        
        results_table.append({
            "Model": name,
            "Precision@3": f"{metrics['precision']:.2f}",
            "Recall@3": f"{metrics['recall']:.2f}",
            "Latency (s)": f"{metrics['latency']:.3f}",
            "Memory (MB)": f"{metrics['memory_mb']:.2f}"
        })

    print("\n\n✅ PHASE 1: RETRIEVAL EVALUATION TABLE ✅")
    print("-" * 75)
    print(f"{'Method':<20} | {'Precision@3':<12} | {'Recall@3':<10} | {'Latency':<10} | {'Memory'}")
    print("-" * 75)
    for row in results_table:
        print(f"{row['Model']:<20} | {row['Precision@3']:<12} | {row['Recall@3']:<10} | {row['Latency (s)'] + 's':<10} | {row['Memory (MB)']} MB")
    print("-" * 75)

    # --- Phase 2: Generation Evaluation ---
    # Ensure your FastAPI server is running before executing this script!
    try:
        # We test the pipeline end-to-end using Gemini as the primary generator
        evaluate_generation(ground_truth_data, provider="gemini", alpha=0.5)
    except requests.exceptions.ConnectionError:
        print("\n❌ API Connection Error: Could not reach FastAPI backend.")
        print("Please ensure you are running 'python -m uvicorn api.main:app' in another terminal before running the generation evaluation.")