import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "indices")

# Chunking Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_STRATEGY = "recursive"

# Retrieval Parameters
DENSE_MODEL = "all-MiniLM-L6-v2"
HYBRID_ALPHA = 0.5
TOP_K = 5