from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from threading import Lock
from .schemas import AnalysisResult

@dataclass
class BatchVideoItem:
    video_id: str
    filename: str
    status: str = "queued"   # queued, processing, done, error
    progress: int = 0
    error: Optional[str] = None
    result: Optional[AnalysisResult] = None

@dataclass
class BatchJob:
    batch_id: str
    items: List[BatchVideoItem] = field(default_factory=list)

class BatchStore:
    def __init__(self):
        self._lock = Lock()
        self._batches: Dict[str, BatchJob] = {}

    def create(self, batch_id: str, items: List[BatchVideoItem]) -> BatchJob:
        with self._lock:
            job = BatchJob(batch_id=batch_id, items=items)
            self._batches[batch_id] = job
            return job

    def get(self, batch_id: str) -> Optional[BatchJob]:
        with self._lock:
            return self._batches.get(batch_id)

BATCH_STORE = BatchStore()
