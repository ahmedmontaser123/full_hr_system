"""
Sets dummy values for required Settings fields so test collection doesn't
need a real .env, database, or Redis instance — the modules under test in
tests/test_pipeline_logic.py never actually use these values (they're pure
functions), but importing them pulls in helpers.config.get_settings(),
which requires these fields to exist.
"""
import os

os.environ.setdefault("APP_NAME", "hireme-test")
os.environ.setdefault("APP_VERSION", "0.0.0-test")

# AI Interview required fields
os.environ.setdefault("ALLOWED_MIME_TYPES", '["audio/mpeg","audio/wav"]')
os.environ.setdefault("MAX_FILE_SIZE_MB", "10")
os.environ.setdefault("WHISPER_MODEL_SIZE", "small")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_COMPUTE_TYPE", "int8")
os.environ.setdefault("LLM_OLLAMA_MODEL", "qwen2.5:3b")

# ATS required fields
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault(
    "FILE_ALLOWED_TYPES",
    '["application/pdf","application/vnd.openxmlformats-officedocument.wordprocessingml.document","application/msword"]',
)
os.environ.setdefault("FILE_MAX_SIZE", "5242880")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
