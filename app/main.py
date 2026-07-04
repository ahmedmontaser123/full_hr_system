import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from helpers import get_settings
from helpers.logger import setup_logger
from helpers.database import close_pool

from routers.base import base_router
from routers.candidates import candidate_router
from routers.jobs import jobs_router
from routers.matching import matching_router

from routers.sessions import router as interview_router


setup_logger()
logger = logging.getLogger("hireme")
settings = get_settings()

# Interview components (lazily initialized)
_llm_provider = None
_chains = None
_whisper = None
_transcript = None


def get_chains():
    global _chains
    if _chains is None:
        from llm.chains import Chains
        from llm.providers.ollama_provider import OllamaProvider
        global _llm_provider
        _llm_provider = OllamaProvider(settings)
        _chains = Chains(_llm_provider.get_llm())
    return _chains


def get_transcript():
    global _transcript
    if _transcript is None:
        from llm.transcript import Transcript
        from llm.providers.faster_whisper_provider import WhisperLoader
        global _whisper
        _whisper = WhisperLoader(settings)
        _transcript = Transcript(_whisper)
    return _transcript


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up ATS pipeline models...")
    try:
        from pipeline.models import ModelRegistry
        ModelRegistry.warm_up()
    except Exception as e:
        logger.warning("Model warm-up skipped (models may not be cached): %s", e)
    yield
    logger.info("Shutting down - closing DB pool...")
    await close_pool()


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- AI Interview state ----------
app.state.sessions: dict = {}
app.state._get_chains = get_chains
app.state._get_transcript = get_transcript

# ---------- Include all routers ----------
app.include_router(interview_router)
app.include_router(base_router)
app.include_router(candidate_router)
app.include_router(jobs_router)
app.include_router(matching_router)

app.mount("/console", StaticFiles(directory="static", html=True), name="console")


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}
