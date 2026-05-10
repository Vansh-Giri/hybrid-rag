import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Base Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    DB_DIR: str = os.path.join(BASE_DIR, "vectorstore")
    EVAL_RESULTS_PATH: str = os.path.join(BASE_DIR, "evaluation", "results.json")

    # API Keys (Loaded automatically from .env)
    GEMINI_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")

    # Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OLLAMA_MODEL: str = "phi4-mini:latest"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"

    # Chunking Parameters
    CHUNK_STRATEGY: str = "semantic"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    SEMANTIC_PERCENTILE_THRESHOLD: int = 90

    # API Server Settings
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # Semantic Cache Settings
    CACHE_INDEX_PATH: str = os.path.join(DB_DIR, "cache.index")
    CACHE_MAP_PATH: str = os.path.join(DB_DIR, "cache_map.json")
    CACHE_SIMILARITY_THRESHOLD: float = 0.95

    # Logging Settings
    LOG_FILE_PATH: str = os.path.join(BASE_DIR, "rag_system.log")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate a global settings object
settings = Settings()