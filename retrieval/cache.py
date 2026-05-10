import os
import sys
import json
import faiss
import numpy as np

# Ensure project root is in path to import logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("Cache")

class SemanticCache:
    def __init__(self, embedder, threshold: float, index_path: str, map_path: str):
        self.embedder = embedder
        self.threshold = threshold
        self.index_path = index_path
        self.map_path = map_path
        
        # We need the dimension size from the model to initialize FAISS
        sample_embedding = self.embedder.encode(["test"])
        self.dimension = sample_embedding.shape[1]

        # Load existing cache or create a new one
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            logger.info(f"Loaded Semantic Cache with {self.index.ntotal} items.")
        else:
            # Use Inner Product for Cosine Similarity (requires normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.mapping = {}

    def check(self, query: str) -> dict:
        """Searches the cache. Returns the cached response dictionary if hit, else None."""
        if self.index.ntotal == 0:
            return None

        # Embed and normalize the query for cosine similarity
        query_emb = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(query_emb)

        # Search top 1
        distances, indices = self.index.search(query_emb, 1)
        similarity_score = distances[0][0]

        if similarity_score >= self.threshold:
            idx = str(indices[0][0])
            logger.info(f"Cache HIT! (Similarity: {similarity_score:.4f})")
            cached_data = self.mapping.get(idx)
            # Tag the response so the UI knows it was cached
            cached_data["answer"] = f"⚡ [CACHED] {cached_data['answer']}"
            return cached_data
            
        logger.info(f"Cache MISS (Highest Similarity: {similarity_score:.4f})")
        return None

    def add(self, query: str, response_dict: dict):
        """Adds a new query and its generated response to the cache."""
        query_emb = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(query_emb)

        idx = self.index.ntotal
        self.index.add(query_emb)
        self.mapping[str(idx)] = response_dict

        # Persist to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=4)