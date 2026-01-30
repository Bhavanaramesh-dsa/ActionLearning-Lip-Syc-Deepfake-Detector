from __future__ import annotations
import os, uuid, subprocess, json, time, hmac, hashlib, base64
from pathlib import Path
from typing import Optional, Dict, Any

# =============================================================================
# PATH CONFIGURATION (LOCAL + DOCKER SAFE)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TMP_DIR = Path(os.getenv("APP_TMP", BASE_DIR / "tmp"))
REPORT_DIR = Path(os.getenv("APP_REPORTS", BASE_DIR / "reports"))
WEB_DIR = Path(os.getenv("APP_WEB", BASE_DIR / "web"))

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# =============================================================================
# DIRECTORY MANAGEMENT
# =============================================================================

def ensure_dirs():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FILE HELPERS
# =============================================================================

def new_video_id() -> str:
    return uuid.uuid4().hex[:16]

def safe_ext(filename: str) -> str:
    return Path(filename).suffix.lower()

def validate_filename(filename: str):
    ext = safe_ext(filename)
    if ext not in ALLOWED_EXT:
        raise ValueError(
            f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXT)}"
        )

def save_upload_to_tmp(video_id: str, filename: str, data: bytes) -> Path:
    ensure_dirs()
    ext = safe_ext(filename) or ".mp4"
    out = TMP_DIR / f"{video_id}{ext}"
    out.write_bytes(data)
    return out

def cleanup_video(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

# =============================================================================
# VIDEO PROBING (FFMPEG / FFPROBE)
# =============================================================================

def ffprobe_json(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "ffprobe failed")
    return json.loads(p.stdout)

def probe_duration_seconds(path: Path) -> float:
    info = ffprobe_json(path)
    fmt = info.get("format", {})
    dur = fmt.get("duration")
    try:
        return float(dur)
    except Exception:
        for s in info.get("streams", []):
            if s.get("duration"):
                try:
                    return float(s["duration"])
                except Exception:
                    pass
    return 0.0

def probe_has_audio(path: Path) -> bool:
    info = ffprobe_json(path)
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))

def trim_to_max_seconds(in_path: Path, out_path: Path, max_s: float) -> bool:
    dur = probe_duration_seconds(in_path)
    if dur <= max_s + 1e-6:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-t", str(max_s),
        "-c", "copy",
        str(out_path)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        # Fallback re-encode
        cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-t", str(max_s),
            "-c:v", "libx264",
            "-c:a", "aac",
            str(out_path)
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip() or "ffmpeg trim failed")

    return True

# =============================================================================
# SIMPLE COOKIE AUTH (SIGNED, NO DB)
# =============================================================================

def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def sign_token(payload: Dict[str, Any], secret: str, ttl_s: int = 60 * 60 * 6) -> str:
    now = int(time.time())
    payload2 = dict(payload)
    payload2["iat"] = now
    payload2["exp"] = now + ttl_s

    data = json.dumps(
        payload2, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    sig = hmac.new(secret.encode("utf-8"), data, hashlib.sha256).digest()
    return _b64(data) + "." + _b64(sig)

def verify_token(token: str, secret: str) -> Optional[Dict[str, Any]]:
    try:
        data_b64, sig_b64 = token.split(".", 1)
        data = _b64d(data_b64)
        sig = _b64d(sig_b64)

        expected = hmac.new(
            secret.encode("utf-8"), data, hashlib.sha256
        ).digest()

        if not hmac.compare_digest(sig, expected):
            return None

        payload = json.loads(data.decode("utf-8"))
        if int(time.time()) > int(payload.get("exp", 0)):
            return None

        return payload
    except Exception:
        return None

def load_users() -> Dict[str, str]:
    raw = os.getenv("APP_USERS_JSON", '{"admin":"admin123"}')
    try:
        return json.loads(raw)
    except Exception:
        return {"admin": "admin123"}