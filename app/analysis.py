from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .schemas import AnalysisResult, Segment
from .utils import probe_duration_seconds, probe_has_audio
from .windowing import split_into_windows
from .quality_check import check_video_quality
from .model_runtime import predict_window, has_checkpoint
from .heatmap import make_timeline_heatmap, make_alignment_curve

UNCERTAIN_LOW = 0.45
UNCERTAIN_HIGH = 0.55
UNCERTAIN_STD_THRESHOLD = 0.15

def _confidence_level(conf: float) -> str:
    if conf >= 0.80:
        return "High"
    if conf >= 0.65:
        return "Medium"
    return "Low"

def _risk_level(conf: float, is_uncertain: bool, reliability_score: float | None) -> str:
    # Conservative: uncertainty or poor input -> higher risk
    if is_uncertain:
        return "High"
    if reliability_score is not None and reliability_score < 50:
        return "High"
    if conf >= 0.80:
        return "Low"
    if conf >= 0.65:
        return "Medium"
    return "High"

def _segments_from_scores(scores: List[float], windows, threshold: float = 0.62) -> List[Segment]:
    segs: List[Segment] = []
    active = None
    for sc, w in zip(scores, windows):
        bad = sc >= threshold
        if bad and active is None:
            active = [w.start, w.end]
        elif bad and active is not None:
            active[1] = w.end
        elif (not bad) and active is not None:
            segs.append(Segment(start=round(active[0], 3), end=round(active[1], 3)))
            active = None
    if active is not None:
        segs.append(Segment(start=round(active[0], 3), end=round(active[1], 3)))
    # Merge tiny gaps
    merged: List[Segment] = []
    for s in segs:
        if not merged:
            merged.append(s)
        else:
            prev = merged[-1]
            if s.start - prev.end <= 0.15:
                merged[-1] = Segment(start=prev.start, end=s.end)
            else:
                merged.append(s)
    return merged

def run_task3_model(video_id: str, filename: str, video_path: Path,
                    window_s: float = 1.0, stride_s: float = 0.5, max_len_s: float = 10.0,
                    trim_notice: str | None = None) -> AnalysisResult:
    duration_s = probe_duration_seconds(video_path)
    has_audio = probe_has_audio(video_path)

    quality = check_video_quality(video_path)

    windows = split_into_windows(duration_s, window_s=window_s, stride_s=stride_s)
    if not windows:
        # Minimal fallback
        windows = split_into_windows(max(duration_s, 1.0), window_s=1.0, stride_s=0.5)

    scores: List[float] = []
    for w in windows:
        scores.append(float(predict_window(video_path, w.start, w.end)))

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    is_ambiguous = (UNCERTAIN_LOW <= mean_score <= UNCERTAIN_HIGH)
    is_inconsistent = (std_score >= UNCERTAIN_STD_THRESHOLD)
    is_uncertain = bool(is_ambiguous or is_inconsistent)

    label_ui = "Uncertain" if is_uncertain else ("Fake" if mean_score >= 0.5 else "Real")

    # Confidence: distance from boundary
    confidence = float(1.0 - min(1.0, abs(mean_score - 0.5) * 2.0))
    confidence_level = _confidence_level(confidence)
    risk_level = _risk_level(confidence, is_uncertain, quality.reliability_score)

    misaligned_segments = _segments_from_scores(scores, windows, threshold=0.62)
    misalignment_ratio = float(np.mean([1.0 if s >= 0.62 else 0.0 for s in scores])) if scores else 0.0
    alignment_stability = "Low" if std_score >= 0.20 else ("Medium" if std_score >= 0.10 else "High")

    heatmap_b64 = make_timeline_heatmap(scores)
    curve_b64 = make_alignment_curve(scores)

    # Explanation
    chk = "loaded" if has_checkpoint() else "not found (demo mode)"
    audio_note = "" if has_audio else " Audio stream was not detected; reliability is reduced."
    explanation = (
        f"The system analyzes audio–visual temporal alignment using sliding windows "
        f"(window={window_s:.1f}s, stride={stride_s:.1f}s) and aggregates the evidence. "
        f"Model checkpoint: {chk}. "
        f"Mean misalignment score={mean_score:.2f}, window disagreement (std)={std_score:.2f}. "
    )
    if label_ui == "Fake":
        explanation += "Sustained misalignment patterns were detected, consistent with lip-sync manipulation."
    elif label_ui == "Real":
        explanation += "Alignment was generally consistent across time, which is typical of authentic speech videos."
    else:
        explanation += "Evidence is ambiguous or inconsistent across windows; the system reports Uncertain to avoid overconfident decisions."
    if misaligned_segments:
        s0 = misaligned_segments[0]
        explanation += f" The earliest highlighted segment is {s0.start:.1f}-{s0.end:.1f}s."
    explanation += audio_note

    return AnalysisResult(
        video_id=video_id,
        filename=filename,
        label_ui=label_ui,
        confidence=round(confidence, 3),
        confidence_level=confidence_level,  # type: ignore
        risk_level=risk_level,              # type: ignore
        mean_score=round(mean_score, 3),
        std_score=round(std_score, 3),
        window_s=window_s,
        stride_s=stride_s,
        duration_s=round(duration_s, 3),
        misaligned_segments=misaligned_segments,
        misalignment_ratio=round(misalignment_ratio, 3),
        alignment_stability=alignment_stability,
        trim_notice=trim_notice,
        quality=quality,
        heatmap_2d_b64png=heatmap_b64,
        alignment_curve_b64png=curve_b64,
        explanation=explanation,
    )
