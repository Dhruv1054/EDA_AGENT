import asyncio
import hashlib
import re
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.graph import eda_graph
from backend.job_store import create_job, get_cached, get_job, set_cache, update_job
from backend.llm_nodes import chat_with_data
from backend.models import ChatRequest, ChatResponse, JobStatus, UploadResponse

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "outputs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_SIZE_MB = 50

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EDA Agent API",
    description="Production-ready AI-powered EDA — FastAPI + LangGraph + LLaMA3.",
    version="3.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate(content: bytes, filename: str) -> str:
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_SIZE_MB} MB limit.")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(415, "Only CSV files are accepted.")
    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "_", Path(filename).stem)
    return f"{safe}.csv"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _serialise(result: dict) -> dict:
    """Strip non-JSON-serialisable objects (DataFrame) from graph state."""
    return {
        "summary":            result.get("summary", {}),
        "missing_values":     result.get("missing_values", {}),
        "fill_logic":         result.get("fill_logic", []),
        "dup_logic":          result.get("dup_logic", {}),
        "duplicates_removed": result.get("duplicate_count", 0),
        "charts_generated":   result.get("charts_generated", []),
        "llm_insights":       result.get("llm_insights", {}),
        "llm_context":        result.get("llm_context", {}),
        "cleaned_csv_path":   result.get("cleaned_csv_path", ""),
    }

# ── Background pipeline task ──────────────────────────────────────────────────

async def _run_pipeline(
    job_id: str, session_id: str, file_path: str, file_hash: str
) -> None:
    try:
        update_job(job_id, status="running", progress=5, stage="Starting pipeline")
        result = await asyncio.to_thread(
            eda_graph.invoke,
            {"file_path": file_path, "session_id": session_id, "job_id": job_id},
        )
        payload = _serialise(result)
        set_cache(file_hash, payload)
        update_job(job_id, status="done", progress=100,
                   stage="Complete", result=payload)
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_csv(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    content    = await file.read()
    safe_name  = _validate(content, file.filename)
    file_hash  = _sha256(content)

    # Return cached result instantly if same file was processed before
    cached = get_cached(file_hash)
    if cached:
        job_id     = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        create_job(job_id, session_id)
        update_job(job_id, status="done", progress=100,
                   stage="Complete (cached)", result=cached)
        return UploadResponse(job_id=job_id, session_id=session_id,
                              message="Result returned from cache.")

    job_id     = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    create_job(job_id, session_id)

    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    save_path.write_bytes(content)

    background_tasks.add_task(
        _run_pipeline, job_id, session_id, str(save_path), file_hash
    )
    return UploadResponse(job_id=job_id, session_id=session_id,
                          message="Pipeline started.")


@app.get("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return JobStatus(
        job_id=job.job_id, status=job.status,
        progress=job.progress, stage=job.stage,
        error=job.error or None,
    )


@app.get("/result/{job_id}")
async def job_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status != "done":
        raise HTTPException(202, f"Job not complete. Status: {job.status}")
    return JSONResponse(content=job.result)


@app.get("/download/{job_id}")
async def download_cleaned_csv(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job.status != "done":
        raise HTTPException(202, "Job not complete.")

    csv_path = job.result.get("cleaned_csv_path")
    if not csv_path or not Path(csv_path).exists():
        raise HTTPException(404, "Cleaned file not found on server.")

    return FileResponse(
        path=csv_path,
        filename="cleaned_dataset.csv",
        media_type="text/csv"
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest):
    """Conversational follow-up with sliding-window memory via LLaMA3."""
    answer = await asyncio.to_thread(
        chat_with_data, body.query, body.dataset_context, body.history
    )
    return ChatResponse(answer=answer, session_id=body.session_id)
