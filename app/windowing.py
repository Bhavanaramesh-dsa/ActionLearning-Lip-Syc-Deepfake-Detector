from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Window:
    start: float
    end: float

def split_into_windows(duration_s: float, window_s: float = 1.0, stride_s: float = 0.5) -> List[Window]:
    if duration_s <= 0:
        return []
    window_s = max(0.2, float(window_s))
    stride_s = max(0.1, float(stride_s))
    windows: List[Window] = []
    t = 0.0
    while t < duration_s:
        start = t
        end = min(duration_s, t + window_s)
        windows.append(Window(start=start, end=end))
        if end >= duration_s:
            break
        t += stride_s
    return windows
