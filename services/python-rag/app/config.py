"""Environment-based configuration for the Python RAG service."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    chroma_db_path: str = "./chroma_db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
