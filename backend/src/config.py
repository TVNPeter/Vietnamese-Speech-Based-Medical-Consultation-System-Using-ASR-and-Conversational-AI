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
    LLAMA_CHAT_TEMPLATE_FILE: str = "templates/qwen3_no_think.jinja"
    LLAMA_REASONING: str = "off"
    LLAMA_REASONING_BUDGET: int = 0
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 220
    LLM_MAX_RESPONSE_CHARS: int = 800

    # Embedding
    EMBEDDING_MODEL_NAME: str = "Dqdung205/medical_vietnamese_embedding"
    EMBEDDING_DEVICE: str = "auto"
    EMBEDDING_BATCH_SIZE: int = 48
    RAG_BACKEND: str = "faiss"
    CHROMA_PERSIST_DIRECTORY: str = "../models/chromadb"
    CHROMA_COLLECTION_NAME: str = "medical_rag_vi"
    BM25_INDEX_PATH: str = "../models/bm25_index.pkl"

    # FAISS
    FAISS_INDEX_PATH: str = "data/faiss_index.faiss"
    DOC_METADATA_PATH: str = "data/doc_metadata.json"
    DOCUMENTS_DIR: str = "documents"
    SEMANTIC_TOP_K: int = 5
    BM25_TOP_K: int = 5
    RAG_TOP_K: int = 5
    RRF_K: int = 60
    RAG_MIN_EVIDENCE_CHUNKS: int = 2
    RAG_MIN_EVIDENCE_SCORE: float = 0.62
    RAG_EVIDENCE_RELEVANCE_FLOOR: float = 0.50
    RAG_TRUSTED_SOURCE_HINTS: str = "dailymed,medlineplus"
    # Synthetic Question/Answer training samples can be useful for experiments,
    # but are not authoritative clinical references. Keep them off by default.
    RAG_ALLOW_QUESTION_ANSWER_SOURCES: bool = False
    # Optional: spend one Tavily request to favor Vietnamese clinical sources
    # even when the local index has a technically sufficient result.
    RAG_PREFER_VIETNAMESE_WEB: bool = False
    RAG_CONTEXT_MAX_CHUNKS: int = 2
    RAG_CONTEXT_CHARS_PER_CHUNK: int = 900

    # Trusted web fallback. It runs only when local evidence is insufficient and
    # a non-empty TAVILY_API_KEY is present in backend/.env.
    TAVILY_ENABLED: bool = True
    TAVILY_API_KEY: str = ""
    TAVILY_MAX_RESULTS: int = 3
    TAVILY_TIMEOUT_SECONDS: float = 12.0
    TAVILY_TRUSTED_DOMAINS: str = (
        "moh.gov.vn,benhvien108.vn,vinmec.com,medlatec.vn,tamanhhospital.vn,"
        "suckhoedoisong.vn,dailymed.nlm.nih.gov,medlineplus.gov,fda.gov,who.int,nhs.uk"
    )
    TAVILY_PREFERRED_DOMAINS: str = (
        "moh.gov.vn,benhvien108.vn,vinmec.com,medlatec.vn,tamanhhospital.vn,"
        "suckhoedoisong.vn"
    )

    # ASR: Wav2Vec2 CTC -> KenLM + drug hotwords -> ViT5 rewrite
    ASR_MODEL_PATH: str = "../models/asr/best_model_hf"
    ASR_KENLM_PATH: str = "models/kenlm/kenlm_vi_medical_4gram.bin"
    ASR_VIT5_MODEL_PATH: str = "../models/kenlm+vit5/vit5_medical_rewrite_stage2_final"
    ASR_HOTWORDS_PATH: str = "../text/drugs.txt"
    ASR_CORPUS_PATH: str = "../text/corpus.txt"
    ASR_BEAM_WIDTH: int = 16
    ASR_BEAM_PRUNE_LOGP: float = -5.0
    ASR_TOKEN_MIN_LOGP: float = -3.0
    ASR_HOTWORD_WEIGHT: float = 15.0
    ASR_MAX_REWRITE_TOKENS: int = 192
    ASR_REQUIRE_KENLM: bool = False
    ASR_USE_GPU: bool = True
    ASR_WAV2VEC2_USE_GPU: bool = True
    ASR_CUDA_DLL_PATH: str = ""
    ASR_CUDNN_DLL_PATH: str = ""

    # TTS
    TTS_MODEL_PATH: str = "models/tts"
    TTS_ENABLED: bool = True
    TTS_VOICE: str = "\u0054\u0072\u00fa\u0063\u0020\u004c\u0079"
    TTS_OUTPUT_FILENAME: str = "vieneu-tts-l-cystine-utf8.wav"

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
