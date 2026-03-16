# test_env.py
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import fitz  # PyMuPDF
import fastapi
import streamlit as st

def run_test():
    print("Testing environment setup...")
    
    # Test dense embedding model load (forces download if not cached)
    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
    
    # Test FAISS
    d = 384 # dimension for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(d)
    print(f"FAISS index created with dimension: {index.d}")
    
    # Test BM25
    corpus = [["hello", "world"], ["hybrid", "rag", "test"]]
    bm25 = BM25Okapi(corpus)
    print(f"BM25 initialized with corpus size: {bm25.corpus_size}")
    
    print("\nAll core dependencies imported and initialized successfully!")

if __name__ == "__main__":
    run_test()