from __future__ import annotations
from pathlib import Path
from typing import List
import os
from .utils import ffprobe_json
from .schemas import VideoQualityMetrics

def check_video_quality(path: Path) -> VideoQualityMetrics:
    info = ffprobe_json(path)
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    issues: List[str] = []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(v.get("width", 0)) if v else 0
    height = int(v.get("height", 0)) if v else 0
    fps = 0.0
    if v and v.get("avg_frame_rate"):
        num, den = v["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    duration_s = float(fmt.get("duration", 0.0) or 0.0)
    total_frames = int(v.get("nb_frames", 0) or 0)
    if total_frames == 0 and duration_s > 0 and fps > 0:
        total_frames = int(duration_s * fps)

    size_bytes = int(fmt.get("size", 0) or 0)
    file_size_mb = size_bytes / (1024 * 1024) if size_bytes else (path.stat().st_size / (1024*1024))
    bitrate = float(fmt.get("bit_rate", 0) or 0.0)
    bitrate_mbps = bitrate / 1e6 if bitrate else 0.0

    if not v:
        issues.append("No video stream detected")
    if not a:
        issues.append("No audio stream detected")

    if width < 320 or height < 240:
        issues.append("Low resolution")
    if fps and fps < 20:
        issues.append("Low FPS")
    if duration_s and duration_s < 1.0:
        issues.append("Very short clip")

    # Score: start 100, penalize
    score = 100.0
    if "No audio stream detected" in issues:
        score -= 40
    if "Low resolution" in issues:
        score -= 20
    if "Low FPS" in issues:
        score -= 15
    if "Very short clip" in issues:
        score -= 10
    if "No video stream detected" in issues:
        score = 0

    score = max(0.0, min(100.0, score))

    if score >= 85:
        grade = "Excellent"
    elif score >= 70:
        grade = "Good"
    elif score >= 50:
        grade = "Fair"
    else:
        grade = "Poor"

    res = f"{width}x{height}" if width and height else "Unknown"
    return VideoQualityMetrics(
        resolution=res,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        total_frames=total_frames,
        file_size_mb=round(file_size_mb, 2),
        bitrate_mbps=round(bitrate_mbps, 2),
        quality_grade=grade,
        reliability_score=round(score, 1),
        issues=issues,
    )
