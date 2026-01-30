# 🎥 Lip-Sync Deepfake Detector  
**Action Learning MVP – Audio-Visual Deepfake Detection**

An AI-powered web application that detects **lip-sync deepfakes** by analyzing **audio–visual temporal misalignment** in videos.  
Built as part of an **Action Learning project**, the system provides **clear decisions, confidence scores, risk levels, visual explanations, batch processing, and an interactive chatbot**.

---

## 🚀 What This Project Does (In Simple Terms)

When a user uploads a video, the system:

1. Breaks the video into small time windows  
2. Checks whether **mouth movements match the spoken audio**  
3. Detects suspicious mismatches that indicate manipulation  
4. Explains **where**, **why**, and **how confident** the decision is  

This makes deepfake detection **transparent, interpretable, and demo-ready**.

---

## ✨ Key Features

### 🔍 Core Detection
- Lip-sync deepfake detection using **temporal window analysis**
- REAL / FAKE / UNCERTAIN classification
- Confidence score and risk level (Low / Medium / High)

### 📊 Visual Explanations
- Timeline heatmap showing misalignment intensity
- Alignment stability curve
- Clickable timestamps to jump to suspicious moments in the video

### 🧪 Quality & Reliability Checks
- Video resolution, FPS, bitrate analysis
- Reliability score to avoid false positives on low-quality inputs
- Automatic trimming of long videos (default: 10 seconds)

### 📁 Batch Processing
- Upload and analyze multiple videos at once
- Live progress tracking per file
- Interactive results dashboard
- Per-video PDF reports

### 💬 AI Chat Assistant
- Ask questions like:
  - *“Which part is misaligned?”*
  - *“Why is this video considered fake?”*
  - *“How reliable is this result?”*
- Answers are grounded in the actual analysis results

---

## 🏗️ System Architecture (High Level)

Video Upload
↓
Quality Check (resolution, fps, reliability)
↓
Temporal Windowing (sliding windows)
↓
Per-Window Model Scoring
↓
Aggregation & Risk Assessment
↓
Visual Explanations + PDF Report + Chatbot


---

## 🧠 Detection Logic (Important for Jury)

- Each video is split into overlapping time windows (e.g., 1.0s window, 0.5s stride)
- Each window is scored for lip-audio mismatch
- Final decision is based on:
  - Mean score
  - Score variance (stability)
  - Input quality

### Conservative “Uncertain” Policy
A video is marked **UNCERTAIN** if:
- The confidence is borderline, **or**
- Window scores strongly disagree

This avoids over-confident false accusations.

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** (Python)
- **PyTorch** (model inference)
- **FFmpeg** (video probing)
- **Pydantic** (data validation)
- **ReportLab** (PDF generation)

### Frontend
- Vanilla **HTML / CSS / JavaScript**
- Interactive dashboard & video player
- Floating chatbot UI

### Deployment Ready
- Docker support
- Hugging Face Spaces compatible

---

## 📂 Project Structure

.
├── app/
│ ├── main.py # FastAPI routes & auth
│ ├── analysis.py # Core detection pipeline
│ ├── model_runtime.py # Model inference logic
│ ├── windowing.py # Temporal segmentation
│ ├── quality_check.py # Video reliability checks
│ ├── heatmap.py # Visual explanations
│ ├── report.py # PDF generation
│ ├── chatbot.py # Explanation assistant
│ └── schemas.py # API data models
│
├── web/
│ └── index.html # Frontend UI
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run_server.py
└── README.md


---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies
bash
pip install -r requirements.txt
### 2️⃣ Start the server
python run_server.py
### 3️⃣ Open in browser
http://localhost:8000
### Default Login
Username: admin
Password: admin123

🎓 Academic Context

This project was developed as part of an Action Learning initiative, focusing on:

Explainable AI

Trustworthy ML systems

Human-centered decision support

Real-world deployment readiness

🌱 Future Improvements

Replace proxy model with fully trained AV deepfake model

Redis / database for multi-user scalability

Model explainability with phoneme-viseme alignment maps

Cloud deployment with GPU acceleration

👤 Author

Bhavana Ramesh
Master’s in Data Science & Analytics
Action Learning Project

⭐ Final Note

This MVP emphasizes clarity, transparency, and user trust — not just accuracy.

If you are a reviewer or jury member:
👉 Upload a video, explore the timeline, and ask the chatbot why a decision was made.

