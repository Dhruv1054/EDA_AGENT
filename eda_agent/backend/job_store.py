"""In-memory job registry + file-level result cache.
For multi-process deployments, swap with Redis.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    status: str = "queued"   # queued | running | done | failed
    progress: int = 0        # 0–100
    stage: str = "Queued"
    result: dict = field(default_factory=dict)
    error: str = ""


_jobs:  dict[str, JobRecord] = {}
_cache: dict[str, dict] = {}   # SHA-256(file bytes) → serialised EDA result
_lock = threading.Lock()


# ── Job CRUD ──────────────────────────────────────────────────────────────────

def create_job(job_id: str, session_id: str) -> JobRecord:
    job = JobRecord(job_id=job_id, session_id=session_id)
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[JobRecord]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)


# ── File-level result cache ────────────────────────────────────────────────────

def get_cached(file_hash: str) -> Optional[dict]:
    return _cache.get(file_hash)


def set_cache(file_hash: str, result: dict) -> None:
    with _lock:
        _cache[file_hash] = result
