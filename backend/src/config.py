"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings

_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # LLM (llama-server)
    LLAMA_SERVER_PATH: str = "llama-b9850-bin-win-vulkan-x64/llama-server.exe"
    LLM_MODEL_PATH: str = "models/qwen3-4b-thinking.gguf"
    LLAMA_SERVER_HOST: str = "127.0.0.1"
    LLAMA_SERVER_PORT: int = 8080
    LLAMA_CTX_SIZE: int = 4096
    LLAMA_GPU_LAYERS: int = 99

    # Embedding
    EMBEDDING_MODEL_NAME: str = "Dqdung205/medical_vietnamese_embedding"

    # FAISS
    FAISS_INDEX_PATH: str = "data/faiss_index.faiss"
    DOC_METADATA_PATH: str = "data/doc_metadata.json"
    DOCUMENTS_DIR: str = "documents"
    SEMANTIC_TOP_K: int = 5
    BM25_TOP_K: int = 5
    RAG_TOP_K: int = 5
    RRF_K: int = 60

    # ASR / TTS
    ASR_MODEL_PATH: str = "models/asr"
    TTS_MODEL_PATH: str = "models/tts"
    TTS_ENABLED: bool = True
    TTS_VOICE: str = "Trúc Ly"

    # HuggingFace
    HF_TOKEN: str = ""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = {"env_file": str(_BACKEND_DIR / ".env"), "extra": "ignore"}

    def resolve(self, relative: str) -> Path:
        """Resolve a relative path against the backend directory."""
        return _BACKEND_DIR / relative


settings = Settings()
