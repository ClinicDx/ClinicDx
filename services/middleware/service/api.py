"""ClinicDx V1 — FastAPI middleware service.

Provides endpoints for:
  POST /cds/generate          — multi-turn KB tool-use CDS generation
  POST /cds/generate_stream   — streaming CDS via SSE
  GET  /cds/health            — CDS health check
  GET  /scribe/manifest       — encounter concept manifest
  POST /scribe/process        — transcription → structured observations
  POST /scribe/process_audio  — audio → observations (direct pipeline)
  POST /scribe/confirm        — POST confirmed FHIR payloads to OpenMRS
  GET  /api/health            — overall middleware health check
"""

import logging
import logging.config
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware

from .concept_extractor import ConceptExtractor
from .scribe_router import router as scribe_router
from .cds_router import router as cds_router


# ── Structured JSON logging ────────────────────────────────────────────────────

def _configure_logging() -> None:
    """Configure all loggers to emit newline-delimited JSON on stdout."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "service"},
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
    # Suppress noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger("middleware")


# ── Trace-ID propagation middleware ───────────────────────────────────────────

class TraceIDMiddleware(BaseHTTPMiddleware):
    """Inject a per-request UUID trace ID into the request state and response headers.

    Every downstream log call should include trace_id from request.state.trace_id.
    The trace ID is also forwarded to upstream services via X-Trace-Id header so
    that log correlation spans the entire middleware → model → KB chain.
    """

    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        t0 = time.time()
        response = await call_next(request)
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        response.headers["X-Trace-Id"] = trace_id
        logger.info(
            "Request completed",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response


# ── Application lifecycle ─────────────────────────────────────────────────────

extractor: Optional[ConceptExtractor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load CIEL mappings for rule-based fallback.  Shutdown: log goodbye."""
    global extractor
    logger.info("Middleware starting", extra={"service": "middleware"})
    extractor = ConceptExtractor()
    try:
        extractor._load_ciel_mappings()
        logger.info("CIEL mappings loaded", extra={"service": "middleware"})
    except Exception as exc:
        logger.warning(
            "CIEL mappings load failed — rule-based fallback unavailable",
            extra={"service": "middleware", "error": str(exc)},
        )
    yield
    logger.info("Middleware shutting down", extra={"service": "middleware"})


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="ClinicDx Middleware",
    description=(
        "Routes clinical queries between the OpenMRS frontend, "
        "the knowledge base, and the llama-server model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(TraceIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scribe_router)
app.include_router(cds_router)


# ── Request / Response models ──────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    text: str
    duration_ms: float


class ExtractRequest(BaseModel):
    text: str
    form_context: Optional[str] = Field(None)
    encounter_history: Optional[list[dict]] = Field(None)


class Observation(BaseModel):
    concept_id: int
    concept_uuid: Optional[str] = None
    label: str
    value: object
    datatype: str
    units: Optional[str] = None
    confidence: float = 0.0


class ExtractResponse(BaseModel):
    observations: list[dict]
    cds_alerts: list[dict] = []
    fallback: Optional[bool] = None
    duration_ms: float = 0.0


class PipelineResponse(BaseModel):
    transcription: str
    observations: list[dict]
    cds_alerts: list[dict] = []
    fallback: Optional[bool] = None
    transcribe_ms: float = 0.0
    extract_ms: float = 0.0
    total_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    extractor_mode: str
    concepts_loaded: dict


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Middleware health check."""
    if extractor is not None and extractor._model is not None:
        mode = "llm"
    elif extractor is not None and extractor._ciel_data is not None:
        mode = "rule_based"
    else:
        mode = "unavailable"

    return HealthResponse(
        status="ok",
        extractor_mode=mode,
        concepts_loaded=extractor.get_ciel_concepts_summary() if extractor else {},
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """Extract structured CIEL observations from transcribed text."""
    if extractor is None:
        raise HTTPException(503, "Concept extractor not initialised")
    if not request.text.strip():
        raise HTTPException(400, "Empty text")

    t0 = time.time()
    if extractor._model is not None:
        result = extractor.extract(
            text=request.text,
            form_context=request.form_context,
            encounter_history=request.encounter_history,
        )
    else:
        result = extractor._rule_based_fallback(request.text)
    duration_ms = (time.time() - t0) * 1000

    return ExtractResponse(
        observations=result.get("observations", []),
        cds_alerts=result.get("cds_alerts", []),
        fallback=result.get("fallback"),
        duration_ms=round(duration_ms, 1),
    )
