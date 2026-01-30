# Lip-Sync Deepfake Detector MVP - Copilot Instructions

## Project Overview
**Lip-Sync Deepfake Detector** is a FastAPI-based web application that analyzes video files to detect audio-visual misalignment (deepfakes). It uses temporal windowing and ML inference to identify fake videos with explanations of suspicious regions.

**Core Value**: Process uploaded videos → extract sliding windows → score misalignment → return confidence, risk level, and visual heatmaps/curves.

## Architecture & Data Flow

### Layer 1: FastAPI Endpoints (app/main.py)
- **Entry Points**: `/api/analyze` (single video), `/api/batch` (bulk processing), `/api/chat` (Q&A chatbot)
- **Auth**: Cookie-based JWT tokens (users.json). `require_user()` enforces session validation
- **Pattern**: Uploads saved to `TMP_DIR`, analysis results stored in `RESULTS` dict (video_id → AnalysisResult), chat sessions in `CHAT_SESSIONS` dict
- **Key Constraint**: `MAX_LEN_S` (default 10.0s) trims oversized videos before processing

### Layer 2: Analysis Pipeline (app/analysis.py → run_task3_model)
1. **Quality Check** (`quality_check.py`): Extracts resolution, fps, bitrate, reliability_score
2. **Windowing** (`windowing.py`): Split video duration into overlapping windows (e.g., 1.0s windows, 0.5s stride)
3. **Scoring** (`model_runtime.py`): Per-window inference → float probability [0.0, 1.0] for "Fake"
4. **Aggregation**: Mean/std of scores determine label (Real/Fake/Uncertain)
   - **Uncertain Logic**: Triggered if mean ∈ [0.45, 0.55] OR std ≥ 0.15 → Conservative approach
   - **Risk Calculation**: Combines confidence + uncertainty + input quality (reliability_score)
5. **Segment Detection** (`_segments_from_scores`): Merges windows with scores ≥0.62 threshold into misaligned_segments, fills gaps ≤0.15s

### Layer 3: Visualization & Output
- **Heatmap** (`heatmap.py`): Timeline bar chart of per-window scores (base64 PNG)
- **Curve** (`heatmap.py`): Stability plot showing score variance across windows
- **Report** (`report.py`): PDF export with analysis summary
- **Schema** (`schemas.py`): `AnalysisResult` pydantic model - single source of truth for response shape

### External Integrations
- **FFmpeg**: Probe video metadata (duration, audio presence, frames)
- **PyTorch + Transformers**: Model inference backend (placeholder: proxy_predict_window uses deterministic hash-based scores for demo)
- **Pillow + ReportLab**: Image/PDF generation
- **Docker**: Multi-stage with ffmpeg, libgl1 system deps

## Key Workflows & Commands

### Local Development
```bash
python run_server.py                    # Start FastAPI dev server on :8000, auto-reload
pip install -r requirements.txt         # Install dependencies
mkdir -p ./tmp ./reports               # Ensure output dirs exist
```

### Docker Deployment
```bash
docker-compose up                       # Builds and runs with volumes for tmp/reports
```

### Configuration via Environment Variables
- `PORT` (default 8000)
- `APP_SECRET` (JWT signing key)
- `APP_MAX_LEN_S` (max video duration before trimming, default 10.0)
- `APP_MODEL_PATH` (checkpoint path; if missing, uses proxy model for demo)
- `APP_TMP`, `APP_REPORTS`, `APP_WEB` (dir paths, defaults to ./tmp, ./reports, ./web)

## Project Conventions & Patterns

### Error Handling
- Use HTTPException(status_code, detail) for API failures (400 bad file, 401 auth, 422 validation)
- Per-file validation in utils.py (validate_filename checks extension, saves with safe uuid prefix)

### Data Serialization
- **Pydantic Models** (schemas.py) enforce strict validation: AnalysisResult, ChatMessage, LoginRequest/Response
- Base64 encoding for binary data (heatmap_2d_b64png, alignment_curve_b64png) in JSON responses
- Segments use float timestamps, rounded to 3 decimals in output

### Threading & State Management
- `BATCH_STORE` uses Lock for thread-safe batch job tracking (dataclass pattern, no ORM)
- In-memory dicts (RESULTS, CHAT_SESSIONS, VIDEO_PATHS) assume single-process dev; scale to Redis for production
- Background inference via threading (not shown; if batch jobs need async, use BackgroundTasks)

### Naming Conventions
- **Functions**: snake_case, prefix with `_` for internal helpers (e.g., `_confidence_level`)
- **Variables**: Scores are floats [0.0, 1.0] representing "fakeness" probability
- **Paths**: Use pathlib.Path, ensure platform-agnostic (no hardcoded "/app/")
- **Config Keys**: UPPERCASE for envvars and module constants

## Common Development Tasks

### Adding a New Endpoint
1. Define Pydantic schema in `schemas.py`
2. Implement handler in `main.py`, use `require_user(req)` for auth
3. Leverage existing analysis flow or create new module under `app/`
4. Test with FastAPI's built-in OpenAPI docs at `/docs`

### Modifying Analysis Logic
- **Thresholds**: Edit constants in `analysis.py` (UNCERTAIN_LOW/HIGH, UNCERTAIN_STD_THRESHOLD, misalignment threshold 0.62)
- **Quality Checks**: Extend `check_video_quality()` in `quality_check.py`
- **Scoring**: Replace `proxy_predict_window()` in `model_runtime.py` with real model checkpoint loading

### Debugging Model Inference
- Check `has_checkpoint()` to see if real model is available
- Proxy model is deterministic (same video + window → same score); use hash key for reproducibility
- Scores persisted in AnalysisResult for UI inspection and report generation

## Critical Files & Their Roles
| File | Purpose |
|------|---------|
| [app/main.py](app/main.py) | FastAPI app, routing, auth |
| [app/analysis.py](app/analysis.py) | Core pipeline: windowing → scoring → aggregation |
| [app/schemas.py](app/schemas.py) | Pydantic models, enforce request/response shape |
| [app/model_runtime.py](app/model_runtime.py) | Model loading & per-window inference |
| [app/chatbot.py](app/chatbot.py) | FAQ-based response generation for result queries |
| [app/utils.py](app/utils.py) | FFmpeg probing, file I/O, JWT tokens |
| [requirements.txt](requirements.txt) | FastAPI, PyTorch, OpenCV, Transformers versions |
| [Dockerfile](Dockerfile) | Python 3.11 slim, ffmpeg system deps |

## Frontend Integration Patterns (web/index.html)

### Authentication & Session Management
- **Token Flow**: `/api/login` sets httpOnly `session` cookie; `/api/me` checks auth status on page load
- **Redirect Logic**: Hidden `loginCard` + shown `appMain` divs; `checkAuth()` runs automatically
- **Protected Requests**: All fetch calls to `/api/*` include cookie by default (fetch sends cookies for same-origin)

### Data Flow: Single Video Analysis
1. User uploads file → `analyzeSingle()` creates FormData with file + window_s + stride_s
2. POST `/api/analyze` returns `AnalysisResult` JSON (includes video_id, scores, segments, base64 images)
3. Frontend renders:
   - **Decision KPIs**: `renderDecision(res)` displays label_ui pill, confidence bar, misalignment_ratio, quality grade
   - **Clickable Segments**: `renderSegments()` creates chips that call `seekTo(playerId, seconds)` to jump video
   - **Charts**: Base64 PNG images (heatmap_2d_b64png, alignment_curve_b64png) embedded as `<img src="data:image/png;base64,..."/>`
   - **Video Stream**: GET `/api/video/{video_id}` returns FileResponse for playback (allows seeking)

### Chatbot Integration
- **Stateful Conversation**: `chatMessages` array in JavaScript persists user/assistant turns
- **Request Format**: POST `/api/chat` with `{video_id, messages}` where messages are ChatMessage objects
- **Response**: Backend returns updated message list; frontend appends and re-renders
- **Grounding**: Chatbot has access to result object (misaligned segments, confidence, quality) and generates context-aware responses

### Batch Processing Pattern
- **Async Workflow**: 
  1. POST `/api/batch/start` with multiple files → returns `batch_id` + file count
  2. Backend spawns worker thread, queues items in BATCH_STORE
  3. Frontend polls `/api/batch/status/{batch_id}` every 1.2s → updates progress % and status ("queued"/"processing"/"done"/"error")
  4. When all items done, table becomes interactive
- **Row Click Handling**: `selectBatchVideo(videoId)` fetches `/api/result/{videoId}` and populates detail panel
- **Export**: Direct link to `/api/report/{video_id}` opens PDF in new tab

### DOM Patterns
- **Tab Navigation**: Buttons toggle `.active` class + toggle `.hidden` on tab divs
- **Status Indicators**: `.overlay .status` positioned absolute (top-left) over video to show "Analyzing…" / "Loading…"
- **Responsive Grid**: `.row` uses `grid-template-columns: 1.2fr 0.8fr` (breaks to 1fr on <980px)
- **Utility Functions**:
  - `$(id)` = `document.getElementById(id)`
  - `api(path, opts)` wraps fetch, parses JSON, throws on non-2xx (includes detail message)
  - `escapeHtml(s)` prevents XSS in chat + table cells

### Video Playback & Seeking
- **Initial Preview**: `URL.createObjectURL(file)` for instant client-side preview during upload
- **Server Stream**: After analysis, switch to `/api/video/{video_id}` for persistent playback (survives page reload)
- **Click-to-Seek**: Segment chips pass floating-point seconds to `seekTo(playerId, start)` → sets `video.currentTime` + calls `.play()`

### Error Handling
- **User Feedback**: Errors caught in try/catch; displayed in `.muted` status spans (e.g., `$("singleStatus").textContent = e.message`)
- **API Errors**: `api()` function extracts `detail` from JSON response or falls back to HTTP status message
- **Graceful Degradation**: Missing charts/segments show `<div class="muted">No data yet</div>` rather than breaking layout

## Testing & Validation Tips
- Use curl/Postman to test `/api/analyze` with a small MP4 file
- Frontend (web/index.html) provides UI for login, video upload, chat with results
- Check `tmp/` and `reports/` directories after runs for artifacts (videos, PDFs, logs)
- Enable reload mode in dev (uvicorn auto-restarts on code changes)
