from __future__ import annotations
from io import BytesIO
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

from .schemas import AnalysisResult


# =============================================================================
# DISCLAIMER (SINGLE SOURCE OF TRUTH)
# =============================================================================

DISCLAIMER = (
    "This system provides a probabilistic assessment of lip-sync manipulation "
    "based on audio–visual temporal alignment. It is intended for research and "
    "demonstration purposes and does not constitute a forensic or legal verdict."
)


# =============================================================================
# TEXT WRAPPING HELPER
# =============================================================================

def wrap_text(text: str, max_len: int):
    words = text.split()
    line, length = [], 0

    for word in words:
        if length + len(word) + 1 > max_len:
            yield " ".join(line)
            line = [word]
            length = len(word)
        else:
            line.append(word)
            length += len(word) + 1

    if line:
        yield " ".join(line)


# =============================================================================
# PDF GENERATOR
# =============================================================================

def generate_pdf(result: AnalysisResult) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # ===== COLOR PALETTE =====
    header_color = colors.HexColor("#2f3a8f")
    text_color = colors.black
    muted_color = colors.HexColor("#555555")
    success = colors.HexColor("#2ecc71")
    warning = colors.HexColor("#f39c12")
    danger = colors.HexColor("#e74c3c")

    y = h - 2.2 * cm

    # ===== PAGE BREAK HELPER =====
    def check_page_break():
        nonlocal y
        if y < 3 * cm:
            c.showPage()
            y = h - 2.2 * cm
            c.setFont("Helvetica", 10)
            c.setFillColor(text_color)

    # ===== TITLE =====
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(header_color)
    c.drawString(2 * cm, y, "Lip-Sync Deepfake Detection Report")

    y -= 0.6 * cm
    c.setLineWidth(1.5)
    c.line(2 * cm, y, w - 2 * cm, y)

    # ===== META INFORMATION =====
    y -= 0.7 * cm
    c.setFont("Helvetica", 9)
    c.setFillColor(muted_color)
    c.drawString(2 * cm, y, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    y -= 0.4 * cm
    c.drawString(2 * cm, y, f"Video ID: {result.video_id}")
    y -= 0.4 * cm
    c.drawString(2 * cm, y, f"Filename: {result.filename}")

    # ===== DECISION BOX =====
    y -= 0.9 * cm
    c.setFont("Helvetica-Bold", 12)

    decision = result.label_ui.upper()

    if decision == "REAL":
        box_color = success
    elif decision == "FAKE":
        box_color = danger
    else:
        box_color = warning

    decision_text = f"DECISION: {decision}"
    text_width = c.stringWidth(decision_text, "Helvetica-Bold", 12)
    box_width = text_width + 20

    c.setFillColor(box_color)
    c.rect(2 * cm, y - 14, box_width, 18, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.drawString(2 * cm + 10, y - 10, decision_text)

    # ===== ANALYSIS METRICS =====
    y -= 1.2 * cm
    check_page_break()

    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Analysis Metrics")

    y -= 0.6 * cm
    c.setFont("Helvetica", 10)

    metrics = [
        f"Confidence Score: {result.confidence:.2%} ({result.confidence_level})",
        f"Risk Level: {result.risk_level} (model-derived)",
        f"Mean Alignment Score: {result.mean_score:.3f}",
        f"Score Stability (σ): {result.std_score:.3f}",
        f"Misalignment Ratio: {result.misalignment_ratio:.1%}",
    ]

    if result.quality:
        metrics.append(
            f"Video Quality: {result.quality.quality_grade} "
            f"(Reliability {result.quality.reliability_score}/100)"
        )

    for m in metrics:
        check_page_break()
        c.drawString(2.4 * cm, y, "• " + m)
        y -= 0.45 * cm

    # ===== SUSPICIOUS SEGMENTS =====
    y -= 0.5 * cm
    check_page_break()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Detected Suspicious Segments")

    y -= 0.6 * cm
    c.setFont("Helvetica", 10)

    if result.misaligned_segments:
        for i, seg in enumerate(result.misaligned_segments[:10], 1):
            check_page_break()
            c.drawString(
                2.4 * cm,
                y,
                f"{i}. {seg.start:.2f}s – {seg.end:.2f}s "
                f"(duration {(seg.end - seg.start):.2f}s)"
            )
            y -= 0.4 * cm
    else:
        c.setFillColor(muted_color)
        c.drawString(2.4 * cm, y, "No significant misalignment detected.")
        c.setFillColor(text_color)

    # ===== EXPLANATION =====
    y -= 0.7 * cm
    check_page_break()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Explanation")

    y -= 0.5 * cm
    c.setFont("Helvetica", 10)

    text_obj = c.beginText(2.4 * cm, y)
    for line in wrap_text(result.explanation, 95):
        if y < 3 * cm:
            c.drawText(text_obj)
            c.showPage()
            y = h - 2.2 * cm
            text_obj = c.beginText(2.4 * cm, y)
            c.setFont("Helvetica", 10)
        text_obj.textLine(line)
        y -= 0.35 * cm

    c.drawText(text_obj)

    # ===== FOOTER DISCLAIMER =====
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(muted_color)
    c.drawString(2 * cm, 1.8 * cm, f"Disclaimer: {DISCLAIMER}")

    # ===== PAGE NUMBER =====
    c.setFont("Helvetica", 8)
    c.drawRightString(w - 2 * cm, 1.5 * cm, "Page 1")

    c.showPage()
    c.save()

    return buf.getvalue()