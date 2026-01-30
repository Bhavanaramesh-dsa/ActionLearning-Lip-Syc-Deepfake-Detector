from __future__ import annotations
from io import BytesIO
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from .schemas import AnalysisResult

DISCLAIMER = (
    "This system provides a probabilistic assessment of lip-sync manipulation "
    "based on audio–visual temporal alignment. It is intended for research and "
    "demonstration purposes and does not constitute a forensic or legal verdict."
)

def generate_pdf(result: AnalysisResult) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    
    # Color scheme
    header_color = colors.HexColor("#7c6cff")
    success_color = colors.HexColor("#3ddc97")
    warning_color = colors.HexColor("#f7b733")
    danger_color = colors.HexColor("#ff5f6d")
    text_color = colors.HexColor("#e7e9f2")
    bg_color = colors.HexColor("#0b1020")

    y = h - 1.2*cm
    
    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(1.5*cm, y, "🎬 Lip-Sync Deepfake Detection Report")
    y -= 0.8*cm
    
    # Horizontal line
    c.setStrokeColor(header_color)
    c.setLineWidth(2)
    c.line(1.5*cm, y, w-1.5*cm, y)
    y -= 0.6*cm
    
    # Video Info Section
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#9aa3c7"))
    c.drawString(1.5*cm, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Video ID: {result.video_id}")
    y -= 0.5*cm
    c.drawString(1.5*cm, y, f"Filename: {result.filename}")
    y -= 0.8*cm
    
    # Decision Summary Box
    c.setFont("Helvetica-Bold", 12)
    decision_text = f"DECISION: {result.label_ui.upper()}"
    decision_box_y = y
    
    # Box background
    if result.label_ui == "Real":
        box_color = success_color
    elif result.label_ui == "Fake":
        box_color = danger_color
    else:
        box_color = warning_color
    
    c.setFillColor(box_color)
    c.rect(1.5*cm, y - 0.5*cm, 4.5*cm, 0.6*cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.drawString(2*cm, y - 0.35*cm, decision_text)
    y -= 0.9*cm
    
    # Key Metrics
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(text_color)
    c.drawString(1.5*cm, y, "Analysis Metrics")
    y -= 0.6*cm
    
    c.setFont("Helvetica", 10)
    metrics = [
        f"✓ Confidence Score: {result.confidence:.2%} ({result.confidence_level})",
        f"⚠ Risk Level: {result.risk_level}",
        f"📊 Mean Alignment Score: {result.mean_score:.3f}",
        f"📈 Score Stability (σ): {result.std_score:.3f}",
        f"🎯 Misalignment Ratio: {result.misalignment_ratio:.1%}",
    ]
    
    if result.quality:
        metrics.append(f"🔍 Video Quality: {result.quality.quality_grade} (Reliability: {result.quality.reliability_score}/100)")
        metrics.append(f"   Resolution: {result.quality.resolution} | FPS: {result.quality.fps:.1f} | Bitrate: {result.quality.bitrate_mbps:.1f} Mbps")
    
    if result.alignment_stability:
        metrics.append(f"🔄 Stability Assessment: {result.alignment_stability}")
    
    for metric in metrics:
        c.drawString(1.8*cm, y, metric)
        y -= 0.45*cm
    
    y -= 0.4*cm
    
    # Misaligned Segments Section
    if result.misaligned_segments and len(result.misaligned_segments) > 0:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1.5*cm, y, f"Detected Suspicious Segments ({len(result.misaligned_segments)})")
        y -= 0.6*cm
        
        c.setFont("Helvetica", 9)
        for i, seg in enumerate(result.misaligned_segments[:15], 1):
            duration = seg.end - seg.start
            c.drawString(1.8*cm, y, f"{i}. [{seg.start:.2f}s - {seg.end:.2f}s] Duration: {duration:.2f}s")
            y -= 0.4*cm
            
            if y < 3.5*cm:
                # Add page break
                c.showPage()
                y = h - 1.5*cm
                c.setFont("Helvetica", 9)
        
        if len(result.misaligned_segments) > 15:
            c.drawString(1.8*cm, y, f"... and {len(result.misaligned_segments) - 15} more segments")
            y -= 0.4*cm
    else:
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#9aa3c7"))
        c.drawString(1.5*cm, y, "✓ No strong misalignment detected (within configured thresholds)")
        y -= 0.6*cm
    
    y -= 0.6*cm
    
    # Explanation Section
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(text_color)
    c.drawString(1.5*cm, y, "Analysis Explanation")
    y -= 0.6*cm
    
    c.setFont("Helvetica", 9)
    text = c.beginText(1.8*cm, y)
    text.setTextOrigin(1.8*cm, y)
    for line in wrap_text(result.explanation, 100):
        text.textLine(line)
        y -= 0.35*cm
    c.drawText(text)
    
    # Footer disclaimer
    y -= 0.8*cm
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.HexColor("#9aa3c7"))
    footer_text = "⚠️ DISCLAIMER: " + DISCLAIMER
    for line in wrap_text(footer_text, 120):
        c.drawString(1.2*cm, y, line)
        y -= 0.3*cm
    
    c.showPage()
    c.save()
    return buf.getvalue()

def wrap_text(s: str, max_len: int):
    words = s.split()
    line=[]
    n=0
    for w in words:
        if n + len(w) + (1 if line else 0) > max_len:
            yield " ".join(line)
            line=[w]
            n=len(w)
        else:
            line.append(w)
            n += len(w) + (1 if line[:-1] else 0)
    if line:
        yield " ".join(line)
