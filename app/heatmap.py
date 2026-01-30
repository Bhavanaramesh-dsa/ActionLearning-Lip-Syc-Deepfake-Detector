from __future__ import annotations
import base64
from io import BytesIO
from typing import List, Tuple
import numpy as np

# Set non-GUI backend BEFORE importing pyplot to avoid macOS threading issues
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _fig_to_b64png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def make_timeline_heatmap(window_scores: List[float], bins: int = 40) -> str:
    if not window_scores:
        window_scores = [0.5]
    xs = np.linspace(0, 1, len(window_scores))
    xbins = np.linspace(0, 1, bins)
    y = np.interp(xbins, xs, window_scores)
    heat = y.reshape(1, -1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(heat, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks([0, bins//4, bins//2, 3*bins//4, bins-1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Alignment Timeline (Higher = More Misalignment)")
    return _fig_to_b64png(fig)

def make_alignment_curve(window_scores: List[float]) -> str:
    if not window_scores:
        window_scores = [0.5]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(window_scores)
    ax.set_title("Window Misalignment Score Curve")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Score (0..1)")
    return _fig_to_b64png(fig)
