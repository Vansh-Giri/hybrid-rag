import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import time
from sentence_transformers import SentenceTransformer
from retrieval.cache import SemanticCache

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # Import centralized configuration
from ingestion.loader import load_directory
from ingestion.cleaner import clean_documents
from ingestion.chunker import process_chunks
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever
from rag.generator import RAGGenerator
from utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger("API_Main")

pipeline_state = {
    "dense": None,
    "sparse": None,
    "hybrid": None,
    "generator": RAGGenerator(
        gemini_model=settings.GEMINI_MODEL, 
        groq_model=settings.GROQ_MODEL,
        ollama_model=settings.OLLAMA_MODEL
    ),
    "embedder": None,  # Shared globally for caching, chunking, and MMR
    "cache": None,     # Global cache instance
    "is_indexed": False
}

# --- Startup Logic (Loads indexes if they exist) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize global embedder
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
    pipeline_state["embedder"] = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    # Initialize Cache
    pipeline_state["cache"] = SemanticCache(
        embedder=pipeline_state["embedder"],
        threshold=settings.CACHE_SIMILARITY_THRESHOLD,
        index_path=settings.CACHE_INDEX_PATH,
        map_path=settings.CACHE_MAP_PATH
    )

    if os.path.exists(os.path.join(settings.DB_DIR, "dense.index")):
        logger.info("Found existing databases on disk. Loading into memory...")
        dense = DenseRetriever(embedder=pipeline_state["embedder"])
        dense.load(settings.DB_DIR)
        
        sparse = SparseRetriever()
        sparse.load(settings.DB_DIR)
        
        pipeline_state["dense"] = dense
        pipeline_state["sparse"] = sparse
        
        # Initialize Hybrid with the embedder for MMR
        pipeline_state["hybrid"] = HybridRetriever(
            dense, 
            sparse, 
            embedder=pipeline_state["embedder"], 
            alpha=0.5
        )
        
        pipeline_state["is_indexed"] = True
        logger.info("Successfully loaded databases! Ready for queries.")
    else:
        logger.warning("No databases found on disk. Please trigger the /index endpoint.")
    yield

app = FastAPI(title="Hybrid RAG API", lifespan=lifespan)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    alpha: float = 0.5
    history: List[ChatMessage] = []
    provider: str = "gemini"
    bypass_cache: bool = False

class SourceItem(BaseModel):
    source: str
    page: Any
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    latency_seconds: float
    used_fallback: bool


@app.post("/index")
def index_documents():
    try:
        logger.info("Starting indexing process...")
        
        raw_docs = load_directory(settings.DATA_DIR)
        if not raw_docs:
            raise HTTPException(status_code=404, detail="No documents found.")
            
        cleaned_docs = clean_documents(raw_docs)
        
        logger.info(f"Processing {settings.CHUNK_STRATEGY} Chunks...")
        chunks = process_chunks(
            cleaned_docs, 
            strategy=settings.CHUNK_STRATEGY, 
            embedder=pipeline_state["embedder"] # Use global embedder here
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks generated. Check document formatting.")

        dense = DenseRetriever(embedder=pipeline_state["embedder"])
        dense.index_documents(chunks)
        dense.save(settings.DB_DIR) 
        
        sparse = SparseRetriever()
        sparse.index_documents(chunks)
        sparse.save(settings.DB_DIR) 
        
        pipeline_state["dense"] = dense
        pipeline_state["sparse"] = sparse
        
        # Initialize Hybrid with the embedder for MMR
        pipeline_state["hybrid"] = HybridRetriever(
            dense, 
            sparse, 
            embedder=pipeline_state["embedder"], 
            alpha=0.5
        )
        
        pipeline_state["is_indexed"] = True
        
        success_msg = f"Successfully processed, indexed, and saved {len(chunks)} {settings.CHUNK_STRATEGY} chunks to disk."
        logger.info(success_msg)
        return {"message": success_msg}
        
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_system(request: QueryRequest):
    if not pipeline_state["is_indexed"]:
        raise HTTPException(status_code=400, detail="System not indexed.")
        
    start_time = time.time()
    cache = pipeline_state["cache"]
    
    # 1. CHECK CACHE FIRST (Only if bypass_cache is False)
    if not request.bypass_cache:
        cached_response = cache.check(request.query)
        if cached_response:
            cached_response["latency_seconds"] = round(time.time() - start_time, 4)
            return QueryResponse(**cached_response)
    
    # 2. FULL RETRIEVAL/GENERATION
    hybrid = pipeline_state["hybrid"]
    generator = pipeline_state["generator"]
    
    hybrid.alpha = request.alpha
    top_chunks = hybrid.search(request.query, top_k=request.top_k)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    answer, used_fallback = generator.generate_answer(
        request.query, 
        top_chunks, 
        history=history_dicts, 
        provider=request.provider
    )
    
    sources = [
        {
            "source": os.path.basename(chunk["metadata"].get("source", "Unknown")),
            "page": chunk["metadata"].get("page", "N/A"),
            "score": round(score, 4)
        }
        for chunk, score in top_chunks
    ]
    
    latency = round(time.time() - start_time, 2)
    
    response_data = {
        "query": request.query, 
        "answer": answer, 
        "sources": sources,
        "latency_seconds": latency,
        "used_fallback": used_fallback
    }
    
    # 3. SAVE RESULT TO CACHE (Only if bypass_cache is False)
    if not request.bypass_cache:
        cache.add(request.query, response_data)
    
    return QueryResponse(**response_data)