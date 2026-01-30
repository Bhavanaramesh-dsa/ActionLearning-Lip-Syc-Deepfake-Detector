from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os, threading, uuid
from typing import Optional, Dict, Any, List
import logging
from collections import deque

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .utils import (
    new_video_id, validate_filename, save_upload_to_tmp,
    probe_duration_seconds, trim_to_max_seconds, cleanup_video,
    WEB_DIR, REPORT_DIR, TMP_DIR,
    sign_token, verify_token, load_users, ensure_dirs
)
from .analysis import run_task3_model
from .report import generate_pdf
from .schemas import AnalysisResult, ChatRequest, ChatResponse, ChatMessage, LoginRequest, LoginResponse
from .chatbot import chat as chat_logic
from .batch import BATCH_STORE, BatchVideoItem

app = FastAPI(title="Lip-Sync Deepfake Detector MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000","http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_dirs()

# In-memory result store (demo)
RESULTS: Dict[str, AnalysisResult] = {}
VIDEO_PATHS: Dict[str, str] = {}  # video_id -> file path
CHAT_SESSIONS: Dict[str, List[ChatMessage]] = {}  # session_key -> messages

APP_SECRET = os.getenv("APP_SECRET", "dev-secret")
MAX_LEN_S = float(os.getenv("APP_MAX_LEN_S", "10.0"))

# In-memory buffer for recent auth-related logs (dev-only, controlled via ENABLE_DEBUG_LOGS=1)
AUTH_LOGS: deque = deque(maxlen=200)

def add_auth_log(level: str, message: str) -> None:
    """Append a sanitized auth-related log entry to the in-memory buffer."""
    # Do not include tokens or sensitive data in these messages.
    ts = datetime.utcnow().isoformat() + "Z"
    AUTH_LOGS.append({"ts": ts, "level": level, "msg": str(message)})

def get_user(req: Request) -> Optional[str]:
    token = req.cookies.get("auth_token")
    logger = logging.getLogger(__name__)
    if not token:
        logger.debug("get_user: no auth_token cookie present")
        add_auth_log("debug", "get_user: no auth_token cookie present")
        return None
    try:
        payload = verify_token(token, APP_SECRET)
    except Exception as e:
        logger.warning("get_user: token verification exception: %s", str(e))
        add_auth_log("warning", f"get_user: token verification exception: {str(e)[:200]}")
        return None
    if not payload:
        logger.debug("get_user: token verification returned falsy payload")
        add_auth_log("debug", "get_user: token verification returned falsy payload")
        return None
    return payload.get("username")

def require_user(req: Request) -> str:
    u = get_user(req)
    if not u:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return u

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = Path("/app/web/index.html")
    if not index_path.exists():
        # dev mode
        index_path = WEB_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@app.post("/api/login", response_model=LoginResponse)
def login(body: LoginRequest):
    users = load_users()
    if users.get(body.username) != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = sign_token({"username": body.username}, APP_SECRET)
    resp = JSONResponse({"ok": True, "username": body.username})
    # For local dev / http://localhost ensure secure=False and path set so browsers accept the cookie
    resp.set_cookie(
    key="auth_token",
    value=token,
    httponly=True,
    secure=False,
    samesite="lax",
    path="/",
    max_age=60 * 60 * 6,   # 🔥 THIS IS THE FIX
)
    logging.getLogger(__name__).info("login: issued auth_token for user %s", body.username)
    add_auth_log("info", f"login: issued auth_token for user {body.username}")
    return resp

@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    # Ensure we delete the same cookie name/path we set during login
    resp.delete_cookie("auth_token", path="/")
    logging.getLogger(__name__).info("logout: cleared auth_token cookie")
    add_auth_log("info", "logout: cleared auth_token cookie")
    return resp

@app.get("/api/me")
def me(req: Request):
    u = get_user(req)
    return {"authenticated": bool(u), "username": u}

@app.get("/debug/logs")
def debug_logs():
    """Return recent auth-related logs (dev-only).
    Enabled when either `ENABLE_DEBUG_LOGS=1` or when APP_SECRET starts with "dev-" (convenient for local dev).
    """
    enabled_env = os.getenv("ENABLE_DEBUG_LOGS", "0") == "1"
    enabled_dev_secret = isinstance(APP_SECRET, str) and APP_SECRET.startswith("dev-")
    if not (enabled_env or enabled_dev_secret):
        raise HTTPException(status_code=404, detail="Not found")
    # Return the recent auth logs without sensitive data
    logging.getLogger(__name__).info("/debug/logs accessed; returning %d entries", len(AUTH_LOGS))
    return {"logs": list(AUTH_LOGS)}
@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(req: Request, file: UploadFile = File(...), window_s: float = Form(1.0), stride_s: float = Form(0.5)):
    require_user(req)

    if not file.filename:
        raise HTTPException(400, "Missing filename")
    validate_filename(file.filename)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    video_id = new_video_id()
    path = save_upload_to_tmp(video_id, file.filename, data)
    VIDEO_PATHS[video_id] = str(path)

    # Trim if needed
    trim_notice = None
    try:
        dur = probe_duration_seconds(path)
        if dur > MAX_LEN_S + 1e-6:
            trimmed = path.with_name(path.stem + "_trimmed" + path.suffix)
            changed = trim_to_max_seconds(path, trimmed, MAX_LEN_S)
            if changed:
                cleanup_video(path)
                path = trimmed
                VIDEO_PATHS[video_id] = str(path)
                trim_notice = f"Video was automatically trimmed to {MAX_LEN_S:.0f}s to match the model input limit."
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}")

    res = run_task3_model(video_id=video_id, filename=file.filename, video_path=path,
                         window_s=float(window_s), stride_s=float(stride_s), max_len_s=MAX_LEN_S,
                         trim_notice=trim_notice)
    RESULTS[video_id] = res

    # init chat session for this video
    CHAT_SESSIONS[video_id] = [
        ChatMessage(role="assistant", content="I can explain the result. Ask: 'Which part is misaligned?' or 'Explain the risk and confidence.'")
    ]
    return res

@app.get("/api/video/{video_id}")
def get_video(req: Request, video_id: str):
    require_user(req)
    p = VIDEO_PATHS.get(video_id)
    if not p or not Path(p).exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(p, media_type="video/mp4")

@app.get("/api/result/{video_id}", response_model=AnalysisResult)
def get_result(req: Request, video_id: str):
    require_user(req)
    if video_id not in RESULTS:
        raise HTTPException(404, "Result not found")
    return RESULTS[video_id]

@app.get("/api/report/{video_id}")
def report(req: Request, video_id: str):
    require_user(req)
    res = RESULTS.get(video_id)
    if not res:
        raise HTTPException(404, "Result not found")
    pdf_bytes = generate_pdf(res)
    out = REPORT_DIR / f"report_{video_id}.pdf"
    out.write_bytes(pdf_bytes)
    return FileResponse(str(out), media_type="application/pdf", filename=out.name)

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: Request, body: ChatRequest):
    require_user(req)
    # Video-scoped chat
    if not body.video_id:
        raise HTTPException(400, "video_id required")
    vid = body.video_id
    res = RESULTS.get(vid)
    msgs = body.messages
    
    try:
        # Keep a server-side session too
        updated = chat_logic(msgs, res)
        CHAT_SESSIONS[vid] = updated
        return ChatResponse(messages=updated)
    except Exception as e:
        import logging
        logging.error(f"Chat error for video {vid}: {str(e)}", exc_info=True)
        # Return error message in chat
        fallback_response = ChatMessage(
            role="assistant",
            content=f"Sorry, I encountered an error processing your question. Error: {str(e)[:100]}"
        )
        return ChatResponse(messages=msgs + [fallback_response])

# ---------------------------
# Batch endpoints (UI dashboard)
# ---------------------------
@app.post("/api/batch/start")
async def batch_start(req: Request, files: List[UploadFile] = File(...), window_s: float = Form(1.0), stride_s: float = Form(0.5)):
    require_user(req)
    if not files:
        raise HTTPException(400, "No files provided")

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[Batch] batch_start: received {len(files)} files")
    
    batch_id = uuid.uuid4().hex[:10]
    items: List[BatchVideoItem] = []
    for f in files:
        if not f.filename:
            continue
        validate_filename(f.filename)
        data = await f.read()
        vid = new_video_id()
        path = save_upload_to_tmp(vid, f.filename, data)
        VIDEO_PATHS[vid] = str(path)
        items.append(BatchVideoItem(video_id=vid, filename=f.filename, status="queued", progress=0))

    logger.info(f"[Batch] batch_start: created {len(items)} items for batch {batch_id}")
    job = BATCH_STORE.create(batch_id, items)

    def worker():
        for item in job.items:
            try:
                item.status = "processing"
                item.progress = 10
                p = Path(VIDEO_PATHS[item.video_id])
                trim_notice = None
                
                # Probe duration
                try:
                    dur = probe_duration_seconds(p)
                    if dur > MAX_LEN_S + 1e-6:
                        item.progress = 20
                        trimmed = p.with_name(p.stem + "_trimmed" + p.suffix)
                        changed = trim_to_max_seconds(p, trimmed, MAX_LEN_S)
                        if changed:
                            cleanup_video(p)
                            p = trimmed
                            VIDEO_PATHS[item.video_id] = str(p)
                            trim_notice = f"Video was automatically trimmed to {MAX_LEN_S:.0f}s to match the model input limit."
                except Exception as probe_err:
                    item.status = "error"
                    item.error = f"Preprocessing failed: {str(probe_err)}"
                    item.progress = 100
                    continue
                
                # Run analysis
                try:
                    item.progress = 40
                    res = run_task3_model(item.video_id, item.filename, p, float(window_s), float(stride_s), MAX_LEN_S, trim_notice)
                    RESULTS[item.video_id] = res
                    item.result = res
                    item.progress = 100
                    item.status = "done"
                except Exception as analysis_err:
                    item.status = "error"
                    item.error = f"Analysis failed: {str(analysis_err)}"
                    item.progress = 100
                    continue
                    
            except Exception as e:
                item.status = "error"
                item.error = f"Unexpected error: {str(e)}"
                item.progress = 100

    # Use non-daemon thread to ensure all videos are processed before shutdown
    worker_thread = threading.Thread(target=worker, daemon=False)
    worker_thread.start()
    return {"batch_id": batch_id, "count": len(job.items)}

@app.get("/api/batch/status/{batch_id}")
def batch_status(req: Request, batch_id: str):
    require_user(req)
    job = BATCH_STORE.get(batch_id)
    if not job:
        raise HTTPException(404, "Batch not found")
    
    items_data = [
        {
            "video_id": it.video_id,
            "filename": it.filename,
            "status": it.status,
            "progress": it.progress,
            "error": it.error,
            "result": it.result.model_dump() if it.result else None,
        } for it in job.items
    ]
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[Batch] batch_status {batch_id}: returning {len(items_data)} items")
    
    return {
        "batch_id": job.batch_id,
        "items": items_data
    }
