import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import load_directory
from ingestion.cleaner import clean_documents
from ingestion.chunker import process_chunks
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever
from rag.generator import RAGGenerator

# Define where to save the databases
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

pipeline_state = {
    "dense": None,
    "sparse": None,
    "hybrid": None,
    "generator": RAGGenerator(model_name="phi4-mini:latest"),
    "is_indexed": False
}

# --- Startup Logic (Loads indexes if they exist) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(os.path.join(DB_DIR, "dense.index")):
        print("Found existing databases on disk. Loading into memory...")
        dense = DenseRetriever()
        dense.load(DB_DIR)
        
        sparse = SparseRetriever()
        sparse.load(DB_DIR)
        
        pipeline_state["dense"] = dense
        pipeline_state["sparse"] = sparse
        pipeline_state["hybrid"] = HybridRetriever(dense, sparse, alpha=0.5)
        pipeline_state["is_indexed"] = True
        print("Successfully loaded databases! Ready for queries.")
    else:
        print("No databases found on disk. Please trigger the /index endpoint.")
    yield

app = FastAPI(title="Hybrid RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceItem(BaseModel):
    source: str
    page: Any
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]

@app.post("/index")
def index_documents():
    try:
        print("Starting indexing process...")
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        
        raw_docs = load_directory(data_dir)
        if not raw_docs:
            raise HTTPException(status_code=404, detail="No documents found.")
            
        cleaned_docs = clean_documents(raw_docs)
        chunks = process_chunks(cleaned_docs, strategy="recursive", chunk_size=500, overlap=50)

        dense = DenseRetriever()
        dense.index_documents(chunks)
        dense.save(DB_DIR) # Save to disk!
        
        sparse = SparseRetriever()
        sparse.index_documents(chunks)
        sparse.save(DB_DIR) # Save to disk!
        
        pipeline_state["dense"] = dense
        pipeline_state["sparse"] = sparse
        pipeline_state["hybrid"] = HybridRetriever(dense, sparse, alpha=0.5)
        pipeline_state["is_indexed"] = True
        
        return {"message": f"Successfully processed, indexed, and saved {len(chunks)} chunks to disk."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_system(request: QueryRequest):
    if not pipeline_state["is_indexed"]:
        raise HTTPException(status_code=400, detail="System not indexed.")
        
    hybrid = pipeline_state["hybrid"]
    generator = pipeline_state["generator"]
    
    top_chunks = hybrid.search(request.query, top_k=request.top_k)
    answer = generator.generate_answer(request.query, top_chunks)
    
    sources = [
        {
            "source": os.path.basename(chunk["metadata"].get("source", "Unknown")),
            "page": chunk["metadata"].get("page", "N/A"),
            "score": round(score, 4)
        }
        for chunk, score in top_chunks
    ]
    
    return QueryResponse(query=request.query, answer=answer, sources=sources)