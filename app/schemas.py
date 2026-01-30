from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

ConfidenceLevel = Literal["Low", "Medium", "High"]
RiskLevel = Literal["Low", "Medium", "High"]
LabelUI = Literal["Real", "Fake", "Uncertain"]

class Segment(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)

class VideoQualityMetrics(BaseModel):
    resolution: str
    width: int
    height: int
    fps: float
    duration_s: float
    total_frames: int
    file_size_mb: float
    bitrate_mbps: float
    quality_grade: str
    reliability_score: float
    issues: List[str] = []

class AnalysisResult(BaseModel):
    video_id: str
    filename: str
    label_ui: LabelUI
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    risk_level: RiskLevel
    mean_score: float = Field(..., ge=0.0, le=1.0)
    std_score: float = Field(..., ge=0.0)
    window_s: float
    stride_s: float
    duration_s: float
    misaligned_segments: List[Segment] = []
    misalignment_ratio: float = Field(..., ge=0.0, le=1.0)
    alignment_stability: str
    trim_notice: Optional[str] = None
    quality: Optional[VideoQualityMetrics] = None
    heatmap_2d_b64png: Optional[str] = None
    alignment_curve_b64png: Optional[str] = None
    explanation: str

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    video_id: Optional[str] = None
    batch_id: Optional[str] = None
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    messages: List[ChatMessage]

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    ok: bool
    username: Optional[str] = None
