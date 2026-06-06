import os
import warnings
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

warnings.filterwarnings("ignore")

# ================= App =================
app = FastAPI(title="Thai Sign Language API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Model Loading =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_sign_model.pkl")

svm_model = None

@app.on_event("startup")
def load_model():
    global svm_model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found at {MODEL_PATH}")
        return
    svm_model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")

# ================= Config (ต้องตรงกับ Train) =================
USE_NORMALIZE = True
HANDS_DIM = 63  # 21 landmarks × 3 (x, y, z)
FEATURE_OUT_DIM = 252  # mean(126) + std(126)

# ================= Schemas =================
class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float

class HandLandmarks(BaseModel):
    landmarks: List[LandmarkPoint]  # 21 points

class PredictRequest(BaseModel):
    hands: List[HandLandmarks]      # 1 or 2 hands
    buffer: Optional[List[List[float]]] = None  # rolling frame buffer

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top3: List[dict]
    feature_dim: int

# ================= Helpers =================
def normalize_hand(landmarks: List[LandmarkPoint]) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    origin = pts[0]
    pts -= origin
    scale = max(float(np.linalg.norm(pts[9])), 1e-6)
    return (pts / scale).reshape(-1)

def extract_features(hands: List[HandLandmarks]) -> np.ndarray:
    feat = []
    for i in range(2):
        if i < len(hands):
            if USE_NORMALIZE:
                feat.extend(normalize_hand(hands[i].landmarks))
            else:
                for p in hands[i].landmarks:
                    feat.extend([p.x, p.y, p.z])
        else:
            feat.extend([0.0] * HANDS_DIM)
    return np.array(feat, dtype=np.float32)

def make_mean_std(buffer: List[List[float]]) -> np.ndarray:
    arr = np.array(buffer, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)

# ================= Routes =================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": svm_model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extract features from current frame
    feat = extract_features(req.hands)

    # Use buffer if provided (mean+std over time window), else use single frame
    if req.buffer and len(req.buffer) >= 5:
        buffer = req.buffer[-60:]  # cap at 60 frames
        buffer.append(feat.tolist())
        X = make_mean_std(buffer).reshape(1, -1)
    else:
        # Fallback: duplicate to create mean+std of single frame
        X = np.concatenate([feat, np.zeros_like(feat)]).reshape(1, -1)

    pred = svm_model.predict(X)[0]
    proba = svm_model.predict_proba(X)[0]
    classes = svm_model.classes_

    top_indices = np.argsort(proba)[::-1][:3]
    top3 = [
        {"label": str(classes[i]), "confidence": round(float(proba[i]) * 100, 1)}
        for i in top_indices
    ]

    return PredictResponse(
        prediction=str(pred),
        confidence=round(float(np.max(proba)) * 100, 1),
        top3=top3,
        feature_dim=X.shape[1],
    )

# ================= Serve Frontend =================
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")

    @app.get("/")
    def root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
