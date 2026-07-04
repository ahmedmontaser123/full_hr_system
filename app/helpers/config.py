from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    # --- AI Interview ---
    ALLOWED_MIME_TYPES: list[str]
    MAX_FILE_SIZE_MB: int
    WHISPER_MODEL_SIZE: str
    WHISPER_DEVICE: str
    WHISPER_COMPUTE_TYPE: str
    LLM_MAX_NEW_TOKENS: int = 250
    LLM_TEMPERATURE: float = 0.6
    LLM_TIMEOUT: int = 300
    LLM_OLLAMA_MODEL: str

    # --- ATS ---
    DATABASE_URL: str
    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int
    FILE_MIN_CHARS: int = 50
    FILE_MAX_CHARS: int = 50000
    FILE_MAX_PAGES: int = 10
    DEFAULT_SHORTLIST_LIMIT: int = 150
    DEFAULT_RERANK_TOP_N: int = 20
    MODELS_CACHE_DIR: str = "~/.cache/huggingface/hub"
    MODELS_LOCAL_ONLY: bool = False
    GLINER_MODEL: str = "urchade/gliner_medium-v2.1"
    EMBEDDER_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    USE_ONNX: bool = True
    GLINER_ONNX_DIR: str = "/home/omar-ahmed/onnx_models/gliner"
    GLINER_ONNX_FILE: str = "model_quantized.onnx"
    BGE_M3_ONNX_DIR: str = "/home/omar-ahmed/onnx_models/bge-m3-int8"
    BGE_M3_ONNX_FILE: str = "model_quantized.onnx"
    RERANKER_ONNX_DIR: str = "/home/omar-ahmed/onnx_models/bge-reranker-int8"
    RERANKER_ONNX_FILE: str = "model_quantized.onnx"
    STORAGE_DIR: str = "storage"
    REDIS_URL: str = "redis://localhost:6379/0"

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }


@lru_cache()
def get_settings():
    return Settings()
