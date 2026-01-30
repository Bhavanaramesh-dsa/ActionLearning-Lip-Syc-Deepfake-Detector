from __future__ import annotations
from typing import Dict, List, Optional
import re
from .schemas import AnalysisResult, ChatMessage

FAQ = [
    ("why fake", "The model observed sustained audio–visual timing inconsistency. Check the highlighted misaligned segments and the timeline heatmap."),
    ("why real", "The model observed consistent temporal alignment between audio and lip motion across most windows."),
    ("why uncertain", "The evidence is ambiguous (score near the decision boundary) or inconsistent across windows. This reduces reliability on real-world videos."),
    ("false positive", "Yes. Real videos can be flagged if audio is noisy, the mouth region is small/blurred, or there is high motion/compression. Uncertainty handling is meant to reduce this risk."),
    ("what is misalignment", "Misalignment means the timing of lip movements does not match the timing of speech sounds. The system highlights time ranges where mismatch is strongest."),
    ("what is deepfake", "A deepfake is a synthetic video where audio and video don't match naturally. This detector focuses on lip-sync manipulation specifically."),
    ("how does it work", "The model uses cross-modal attention to compare audio (via Wav2Vec2) and visual (ResNet) features. Sustained misalignment → higher fake score."),
    ("what does confidence mean", "Confidence (0-1) is the model's certainty about the prediction. High confidence + high score = likely fake. Low confidence = uncertain, needs review."),
    ("what is risk level", "Risk combines confidence, uncertainty, and input quality. High risk = be cautious. Low risk = more reliable prediction."),
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def generate_reply(user_text: str, result: Optional[AnalysisResult]) -> str:
    t = _norm(user_text)

    # Greeting detection - only respond if ONLY greeting words
    greetings = ["hello", "hi", "hey"]
    if any(g in t for g in greetings) and len(t) < 10:  # Only short greetings
        return (
            "👋 Hi! I'm your deepfake detection assistant. Ask me about:\n"
            "• Confidence score & reliability\n"
            "• Which parts are misaligned\n"
            "• Why it's marked real/fake\n"
            "• What the risk level means"
        )

    # CONFIDENCE - matches: confidence, explain confidence, what's confidence, what is confidence, understand confidence
    if any(word in t for word in ["confidence", "confident", "score", "certainty"]):
        if result:
            return (
                f"📊 **Confidence Score:** {result.confidence:.2f} ({result.confidence_level})\n\n"
                f"This means the model is {result.confidence:.0%} certain about its prediction.\n"
                f"**Decision:** {result.label_ui}\n"
                f"**Risk Level:** {result.risk_level}\n\n"
                f"High confidence + high score = likely fake.\n"
                f"Low confidence = ambiguous, needs careful review."
            )
        return "Confidence is the model's certainty (0-1) about whether a video is real or fake. High confidence means the model is very sure about its prediction."

    # MISALIGNMENT / SEGMENTS - matches: where, which part, timestamp, segment, misaligned
    if any(word in t for word in ["where", "which", "part", "timestamp", "time", "segment", "misalign"]):
        if result and result.misaligned_segments:
            segs = "\n".join([f"  • {s.start:.1f}–{s.end:.1f}s" for s in result.misaligned_segments[:5]])
            extra = f"\n  ... and {len(result.misaligned_segments) - 5} more" if len(result.misaligned_segments) > 5 else ""
            return f"🎯 **Suspicious Segments:**\n{segs}{extra}\n\nClick these timestamps in the timeline to jump there. Check if lip movements match the audio."
        return "✅ No strong misaligned segments detected. The audio and video appear well-synchronized."

    # RELIABILITY / QUALITY - matches: reliable, reliability, quality, trustworthy
    if any(word in t for word in ["reliable", "reliability", "quality", "trustworthy", "trust"]):
        if result:
            quality_info = ""
            if result.quality:
                try:
                    quality_info = (
                        f"\n🔍 **Video Quality:**\n"
                        f"  Grade: {result.quality.quality_grade}\n"
                        f"  Reliability: {result.quality.reliability_score}/100"
                    )
                    if hasattr(result.quality, 'resolution') and result.quality.resolution:
                        quality_info += f"\n  Resolution: {result.quality.resolution}"
                    if hasattr(result.quality, 'fps') and result.quality.fps:
                        quality_info += f"\n  FPS: {result.quality.fps:.1f}"
                    if hasattr(result.quality, 'bitrate_mbps') and result.quality.bitrate_mbps:
                        quality_info += f"\n  Bitrate: {result.quality.bitrate_mbps:.1f} Mbps"
                except Exception as e:
                    quality_info = f"\n🔍 Quality: {result.quality.quality_grade} (Reliability: {result.quality.reliability_score}/100)"
            
            return (
                f"📋 **Is This Reliable?**\n"
                f"Decision: **{result.label_ui}**\n"
                f"Confidence: **{result.confidence:.2%}**\n"
                f"Risk Level: **{result.risk_level}**{quality_info}\n\n"
                f"Higher reliability = more trustworthy prediction."
            )
        return "Reliability depends on video quality (resolution, FPS, bitrate). Noisy or compressed videos are harder to analyze accurately."

    # RISK LEVEL - matches: risk, risky, dangerous
    if any(word in t for word in ["risk", "risky"]):
        if result:
            return (
                f"⚠️ **Risk Level:** {result.risk_level}\n\n"
                f"Risk combines:\n"
                f"• Confidence in the prediction\n"
                f"• Consistency across the video\n"
                f"• Input video quality\n\n"
                f"**Low Risk:** More reliable prediction.\n"
                f"**High Risk:** Be cautious, might need manual review."
            )
        return "Risk level indicates how cautious you should be. High risk = prediction might be unreliable. Low risk = prediction is more trustworthy."

    # WHY IS IT FAKE/REAL - matches: why, explain, reason
    if any(word in t for word in ["why", "explain", "reason", "because"]):
        if result:
            explanation = result.explanation if result.explanation else "Check the misaligned segments and heatmap."
            return f"📝 **Explanation:**\n{explanation}"
        return "Check the misaligned segments and the timeline heatmap to see where inconsistencies occur."

    # Simple FAQ matching - only if key words are present
    for key, ans in FAQ:
        key_words = key.split()
        if all(w in t for w in key_words):
            return ans

    # Grounded summary if result exists and question is unclear
    if result:
        seg_info = ""
        if result.misaligned_segments:
            seg_info = f"\n📍 First suspicious segment: {result.misaligned_segments[0].start:.1f}–{result.misaligned_segments[0].end:.1f}s"
        return (
            f"📋 **Current Analysis:**\n"
            f"Decision: **{result.label_ui}**\n"
            f"Confidence: **{result.confidence:.2%}**\n"
            f"Risk Level: **{result.risk_level}**{seg_info}\n\n"
            "Try asking:\n• 'What's the confidence?'\n• 'Which part is misaligned?'\n• 'Is this reliable?'"
        )

    return (
        "❓ I didn't quite understand. Try asking:\n"
        "• 'Why is it fake?'\n"
        "• 'Which part is misaligned?'\n"
        "• 'What's the confidence?'\n"
        "• 'Is this reliable?'"
    )

def chat(messages: List[ChatMessage], result: Optional[AnalysisResult]) -> List[ChatMessage]:
    """Multi-turn conversation with context awareness"""
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if not last_user:
        return messages + [ChatMessage(role="assistant", content="How can I help you analyze this video?")]
    
    reply = generate_reply(last_user.content, result)
    return messages + [ChatMessage(role="assistant", content=reply)]
