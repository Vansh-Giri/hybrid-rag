import os
import requests
import json
from typing import List, Dict, Tuple

class RAGGenerator:
    def __init__(self, model_name: str = "phi4-mini:latest", base_url: str = "http://localhost:11434"):
        """
        Initializes the RAG generator using a local Ollama instance.
        Ensure Ollama is running and the model is pulled.
        """
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"
        
    def _build_prompt(self, query: str, context_chunks: List[Tuple[Dict, float]]) -> str:
        """Constructs the prompt template using retrieved context."""
        
        # Combine the text from the top retrieved chunks
        context_text = "\n\n---\n\n".join(
            [f"Source: {chunk['metadata'].get('source', 'Unknown')} (Page {chunk['metadata'].get('page', 'N/A')})\n{chunk['text']}" 
             for chunk, _ in context_chunks]
        )
        
        # Standard RAG Prompt Template
        prompt = f"""You are a helpful and precise technical assistant. 
Use the following retrieved context to answer the user's question. 
If the answer is not contained within the context, say "I don't have enough information in the provided documents to answer that." 
Do not hallucinate or use outside knowledge.

Context:
{context_text}

Question: {query}

Answer:"""
        return prompt

    def generate_answer(self, query: str, context_chunks: List[Tuple[Dict, float]]) -> str:
        """Sends the prompt to the local Ollama LLM and returns the generated answer."""
        if not context_chunks:
            return "No relevant context found to answer the query."

        prompt = self._build_prompt(query, context_chunks)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.1,
                "stop": ["\nQuestion:", "Question:"]
            }
        }

        print(f"Sending prompt to local Ollama ({self.model_name})...")
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Error: No response generated.")
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Is the Ollama service running locally?"
        except Exception as e:
            return f"Error during generation: {str(e)}"

if __name__ == "__main__":
    # Test Full QA Pipeline
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.loader import load_directory
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import process_chunks
    from retrieval.dense import DenseRetriever
    from retrieval.sparse import SparseRetriever
    from retrieval.hybrid import HybridRetriever

    print("--- Booting up Full Pipeline ---")
    
    # 1. Ingestion
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_docs = load_directory(test_dir)
    cleaned_docs = clean_documents(raw_docs)
    chunks = process_chunks(cleaned_docs, strategy="recursive", chunk_size=500, overlap=50)

    # 2. Retrieval
    dense = DenseRetriever()
    dense.index_documents(chunks)
    sparse = SparseRetriever()
    sparse.index_documents(chunks)
    hybrid = HybridRetriever(dense, sparse, alpha=0.5)

    # 3. Generation
    generator = RAGGenerator(model_name="phi4-mini:latest") 
    
    test_query = "What are the objectives of this hybrid RAG project?"
    print(f"\nUser Query: '{test_query}'")
    
    # Retrieve top 3 chunks
    top_chunks = hybrid.search(test_query, top_k=3)
    
    # Generate Answer
    answer = generator.generate_answer(test_query, top_chunks)
    
    print("\n--- Generated Answer ---")
    print(answer)