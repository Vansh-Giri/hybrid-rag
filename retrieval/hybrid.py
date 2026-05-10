from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import util

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, embedder=None, alpha: float = 0.5, rrf_k: int = 60, lambda_mult: float = 0.5):
        """
        Initializes the Hybrid Retriever with RRF and MMR.
        :param dense_retriever: Instance of DenseRetriever
        :param sparse_retriever: Instance of SparseRetriever
        :param embedder: SentenceTransformer instance required for MMR diversity calculations
        :param alpha: Weight for dense vs sparse retrieval
        :param rrf_k: RRF smoothing constant
        :param lambda_mult: MMR parameter (1.0 = Max Relevance, 0.0 = Max Diversity, 0.5 = Balanced)
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.embedder = embedder
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.lambda_mult = lambda_mult

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        # 1. Fetch a larger candidate pool for MMR to process
        fetch_k = max(top_k * 4, 15) 
        
        dense_results = self.dense.search(query, top_k=fetch_k)
        sparse_results = self.sparse.search(query, top_k=fetch_k)
        
        fused_scores = {}
        chunk_map = {}
        
        # 2. Calculate Weighted Reciprocal Rank Fusion (RRF)
        for rank, (chunk, _) in enumerate(dense_results):
            chunk_id = chunk["metadata"].get("chunk_id", hash(chunk["text"]))
            chunk_map[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (self.alpha / (self.rrf_k + rank + 1))
            
        for rank, (chunk, _) in enumerate(sparse_results):
            chunk_id = chunk["metadata"].get("chunk_id", hash(chunk["text"]))
            chunk_map[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + ((1.0 - self.alpha) / (self.rrf_k + rank + 1))
            
        # Sort RRF candidates
        sorted_candidates = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        candidate_chunks = [(chunk_map[cid], score) for cid, score in sorted_candidates[:fetch_k]]

        # 3. Apply Maximal Marginal Relevance (MMR)
        if self.embedder is None or len(candidate_chunks) <= top_k:
            return candidate_chunks[:top_k]

        return self._apply_mmr(candidate_chunks, top_k)

    def _apply_mmr(self, candidate_chunks: List[Tuple[Dict, float]], top_k: int) -> List[Tuple[Dict, float]]:
        texts = [chunk["text"] for chunk, _ in candidate_chunks]
        raw_scores = [score for _, score in candidate_chunks]

        # Normalize RRF scores to [0, 1] for fair comparison with cosine similarity
        min_score, max_score = min(raw_scores), max(raw_scores)
        if max_score > min_score:
            norm_scores = [(s - min_score) / (max_score - min_score) for s in raw_scores]
        else:
            norm_scores = [1.0] * len(raw_scores)

        # Compute embeddings for candidates to check their similarity to each other
        embeddings = self.embedder.encode(texts, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

        selected_indices = []
        unselected_indices = list(range(len(texts)))

        # Select the item with the highest RRF score first
        first_idx = int(np.argmax(norm_scores))
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)

        # Iteratively select chunks balancing Relevance and Diversity
        while len(selected_indices) < top_k and unselected_indices:
            best_score = -np.inf
            best_idx = -1

            for idx in unselected_indices:
                max_sim_to_selected = max([sim_matrix[idx][s_idx] for s_idx in selected_indices])
                
                # MMR Equation
                mmr_score = self.lambda_mult * norm_scores[idx] - (1.0 - self.lambda_mult) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        return [candidate_chunks[i] for i in selected_indices]