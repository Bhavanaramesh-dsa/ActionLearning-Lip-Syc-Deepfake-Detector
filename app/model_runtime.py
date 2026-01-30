from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import cv2
import librosa
import soundfile as sf
from torchvision import transforms

logger = logging.getLogger(__name__)

# ===== Model Path Configuration =====
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATHS = [
    Path(os.getenv("APP_MODEL_PATH")) if os.getenv("APP_MODEL_PATH") else None,
    BASE_DIR / "task3_alignment_model.pt",
    Path("/app/task3_alignment_model.pt"),
]
MODEL_PATH = next((p for p in MODEL_PATHS if p and p.exists()), MODEL_PATHS[1])

# ===== Device Configuration =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Model Constants =====
TARGET_SAMPLE_RATE = 16000  # Audio: 16 kHz
TARGET_VIDEO_FPS = 25  # Video frame rate
MOUTH_ROI_SIZE = (112, 112)  # Mouth ROI frame size (H, W)
AUDIO_SEGMENT_LENGTH = 16000 * 2  # 2 seconds at 16kHz

# ===== Preprocessing Constants =====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ===== Global Model Cache =====
_model = None
_model_loaded = False


def has_checkpoint() -> bool:
    """Check if model weights file exists."""
    return MODEL_PATH.exists()


def get_model() -> Optional[nn.Module]:
    """
    Load and cache the model from checkpoint.
    Returns None if checkpoint doesn't exist.
    """
    global _model, _model_loaded
    
    if _model_loaded:
        return _model
    
    _model_loaded = True
    
    if not has_checkpoint():
        logger.warning(f"Model checkpoint not found at {MODEL_PATH}. Using proxy model.")
        return None
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Handle both dict and model format
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Standard checkpoint format with state_dict + model
                _model = checkpoint.get("model", None)
                if _model is None:
                    logger.warning("Checkpoint has state_dict but no model architecture.")
                    return None
                _model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                # Alternative format - requires model class to be instantiated
                logger.warning("State-dict-only checkpoint. Ensure model class is instantiated.")
                return None
            else:
                # Try to use entire dict as state (unlikely but handle gracefully)
                logger.warning("Unrecognized checkpoint format. Trying direct load...")
                return None
        else:
            # Assume direct model save
            _model = checkpoint
        
        _model.eval()
        _model.to(DEVICE)
        logger.info(f"✓ Model loaded successfully on {DEVICE}")
        return _model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}. Using proxy model.")
        _model = None
        return None


# ===== Audio Processing =====
def extract_audio_segment(video_path: Path, start_s: float, end_s: float) -> torch.Tensor:
    """
    Extract and resample audio segment from video.
    
    Args:
        video_path: Path to video file
        start_s: Start time in seconds
        end_s: End time in seconds
    
    Returns:
        Audio waveform tensor, shape [1, num_samples]
    """
    try:
        # Use ffmpeg via librosa to extract audio
        # librosa.load handles all audio formats
        audio, sr = librosa.load(str(video_path), sr=None, mono=True)
        
        # Convert time to samples
        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        
        # Extract segment
        audio_segment = audio[start_sample:end_sample]
        
        # Resample to target rate if needed
        if sr != TARGET_SAMPLE_RATE:
            audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_segment).float()
        audio_tensor = audio_tensor.unsqueeze(0)  # [1, T]
        
        return audio_tensor
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        raise


def preprocess_audio(audio_waveform: torch.Tensor) -> torch.Tensor:
    """
    Preprocess audio waveform for model input.
    - Normalize to [-1, 1]
    - Ensure correct shape
    
    Args:
        audio_waveform: Raw audio tensor, shape [1, T] or [T]
    
    Returns:
        Preprocessed audio tensor
    """
    if audio_waveform.dim() == 1:
        audio_waveform = audio_waveform.unsqueeze(0)
    
    # Normalize: divide by max absolute value
    max_val = audio_waveform.abs().max()
    if max_val > 0:
        audio_waveform = audio_waveform / (max_val + 1e-8)
    
    return audio_waveform.to(DEVICE)


# ===== Video Processing =====
def extract_mouth_frames(video_path: Path, start_s: float, end_s: float) -> torch.Tensor:
    """
    Extract mouth ROI frames from video segment.
    
    Args:
        video_path: Path to video file
        start_s: Start time in seconds
        end_s: End time in seconds
    
    Returns:
        Frame tensor, shape [num_frames, 3, H, W]
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Calculate frame indices for time window
        start_frame = int(start_s * fps)
        end_frame = int(end_s * fps)
        
        # Ensure indices are within bounds
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract mouth ROI (center crop approximation)
            # TODO: Replace with MediaPipe/dlib face detection for robust mouth ROI extraction
            mouth_roi = extract_mouth_roi_simple(frame)
            frames.append(mouth_roi)
        
        cap.release()
        
        if not frames:
            raise RuntimeError(f"No frames extracted from {start_s}-{end_s}s window")
        
        # Stack frames into tensor [num_frames, 3, H, W]
        frames_tensor = torch.stack(frames)
        return frames_tensor
        
    except Exception as e:
        logger.error(f"Video frame extraction failed: {e}")
        raise


def extract_mouth_roi_simple(frame: np.ndarray) -> torch.Tensor:
    """
    Extract mouth ROI from frame (simple center crop).
    
    TODO: Replace with MediaPipe/dlib for robust face detection.
    
    Args:
        frame: Input frame in BGR format, shape [H, W, 3]
    
    Returns:
        Mouth ROI tensor, shape [3, 112, 112]
    """
    h, w = frame.shape[:2]
    
    # Simple center crop (approximation)
    # In production, use face detection + mouth landmark
    center_y, center_x = h // 2, w // 2
    roi_h, roi_w = int(h * 0.3), int(w * 0.3)
    
    y_start = max(0, center_y - roi_h // 2)
    y_end = min(h, center_y + roi_h // 2)
    x_start = max(0, center_x - roi_w // 2)
    x_end = min(w, center_x + roi_w // 2)
    
    mouth_roi = frame[y_start:y_end, x_start:x_end, :]
    
    # Convert BGR to RGB and resize
    mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB)
    mouth_roi = cv2.resize(mouth_roi, MOUTH_ROI_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor [3, H, W]
    mouth_roi = torch.from_numpy(mouth_roi).float()
    mouth_roi = mouth_roi.permute(2, 0, 1)  # HWC -> CHW
    
    return mouth_roi


def preprocess_video(video_frames: torch.Tensor) -> torch.Tensor:
    """
    Preprocess video frames for model input.
    - Normalize using ImageNet statistics
    - Ensure correct shape
    
    Args:
        video_frames: Raw frame tensor, shape [num_frames, 3, H, W]
    
    Returns:
        Preprocessed frames tensor
    """
    # Normalize to [0, 1] if in [0, 255]
    if video_frames.max() > 1.0:
        video_frames = video_frames / 255.0
    
    # Apply ImageNet normalization
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
    
    video_frames = normalize(video_frames)
    return video_frames.to(DEVICE)


# ===== Model Inference =====
def predict_window(video_path: Path, start_s: float, end_s: float) -> float:
    """
    Predict alignment score for a video window.
    
    Args:
        video_path: Path to video file
        start_s: Window start time in seconds
        end_s: Window end time in seconds
    
    Returns:
        Float probability [0.0, 1.0] for "Fake" (misalignment)
    """
    model = get_model()
    
    if model is not None:
        return _predict_with_model(model, video_path, start_s, end_s)
    else:
        return proxy_predict_window(video_path, start_s, end_s)


def _predict_with_model(model: nn.Module, video_path: Path, start_s: float, end_s: float) -> float:
    """
    Full inference pipeline: extract → preprocess → model → sigmoid.
    
    Args:
        model: Loaded PyTorch model
        video_path: Path to video file
        start_s: Window start time in seconds
        end_s: Window end time in seconds
    
    Returns:
        Probability of lip-sync manipulation
    """
    try:
        logger.debug(f"Running inference on {video_path.name} [{start_s:.2f}–{end_s:.2f}s]")
        
        # 1. Extract audio segment
        audio_waveform = extract_audio_segment(video_path, start_s, end_s)
        
        # 2. Extract video frames (mouth ROI)
        mouth_frames = extract_mouth_frames(video_path, start_s, end_s)
        
        # 3. Preprocess
        audio_waveform = preprocess_audio(audio_waveform)
        mouth_frames = preprocess_video(mouth_frames)
        
        # 4. Model inference
        with torch.no_grad():
            logits = model(audio_waveform, mouth_frames)
        
        # 5. Convert logits to probability via sigmoid
        prob = torch.sigmoid(logits).squeeze().item()
        prob = float(np.clip(prob, 0.0, 1.0))
        
        logger.debug(f"Inference result: {prob:.4f}")
        return prob
        
    except Exception as e:
        logger.error(f"Model inference failed: {e}. Falling back to proxy.")
        return proxy_predict_window(video_path, start_s, end_s)


def proxy_predict_window(video_path: Path, start_s: float, end_s: float) -> float:
    """
    Deterministic fallback model for demo/testing when real model unavailable.
    Produces stable output for same video and window times.
    
    Args:
        video_path: Path to video file
        start_s: Window start time in seconds
        end_s: Window end time in seconds
    
    Returns:
        Stable pseudo-random probability [0.0, 1.0]
    """
    key = f"{video_path.name}:{start_s:.2f}:{end_s:.2f}"
    h = abs(hash(key)) % 10_000
    base = (h / 10_000.0)
    # Pull towards center to allow Uncertain classification
    score = 0.35 + 0.3 * base
    return float(np.clip(score, 0.0, 1.0))


