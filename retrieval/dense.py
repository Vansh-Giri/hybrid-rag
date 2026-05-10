import os
import pickle
import numpy as np
import faiss

class DenseRetriever:
    def __init__(self, embedder=None):
        self.embedder = embedder
        self.index = None
        self.chunk_map = {}
        self.dimension = None

    def index_documents(self, chunks):
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        
        # Generate and normalize embeddings for Cosine Similarity (Inner Product)
        embeddings = self.embedder.encode(texts, convert_to_tensor=False).astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.dimension = embeddings.shape[1]
        num_vectors = embeddings.shape[0]

        # FAISS IVFFlat requires at least 39 * nlist training points.
        # If the dataset is small, fall back to exact search.
        if num_vectors < 1000:
            print(f"Dataset size ({num_vectors}) too small for IVFFlat. Using IndexFlatIP.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
        else:
            # Dynamic calculation of Voronoi cells
            nlist = int(np.sqrt(num_vectors))
            print(f"Initializing IVFFlat with nlist={nlist}...")
            
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            print("Training IVFFlat index...")
            self.index.train(embeddings)
            self.index.add(embeddings)

        # Store metadata mapping
        for i, chunk in enumerate(chunks):
            # Initialize metadata safely if it somehow got lost
            if "metadata" not in chunk:
                chunk["metadata"] = {}
                
            # Only inject a fallback chunk_id if the chunker didn't provide one
            if "chunk_id" not in chunk["metadata"]:
                chunk["metadata"]["chunk_id"] = str(i)
                
            self.chunk_map[i] = chunk

    def search(self, query: str, top_k: int = 3):
        if not self.index or not self.embedder:
            return []
            
        query_emb = self.embedder.encode([query], convert_to_tensor=False).astype('float32')
        faiss.normalize_L2(query_emb)

        # Tune nprobe dynamically at search time to balance speed/recall
        if isinstance(self.index, faiss.IndexIVF):
            nprobe = max(1, self.index.nlist // 10)  # Search 10% of clusters
            self.index.nprobe = nprobe

        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for j, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.chunk_map:
                results.append((self.chunk_map[idx], float(distances[0][j])))
                
        return results

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self.index:
            faiss.write_index(self.index, os.path.join(save_dir, "dense.index"))
        with open(os.path.join(save_dir, "dense_map.pkl"), "wb") as f:
            pickle.dump(self.chunk_map, f)

    def load(self, load_dir: str):
        index_path = os.path.join(load_dir, "dense.index")
        map_path = os.path.join(load_dir, "dense_map.pkl")
        
        if os.path.exists(index_path) and os.path.exists(map_path):
            self.index = faiss.read_index(index_path)
            with open(map_path, "rb") as f:
                self.chunk_map = pickle.load(f)