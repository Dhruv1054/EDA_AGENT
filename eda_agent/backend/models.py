from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    session_id: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str       # queued | running | done | failed
    progress: int     # 0–100
    stage: str
    error: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    session_id: str
    history: list[dict]
    dataset_context: dict


class ChatResponse(BaseModel):
    answer: str
    session_id: str
