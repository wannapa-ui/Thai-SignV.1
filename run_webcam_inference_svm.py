import os
import sys
import cv2
import time
import numpy as np
import warnings
import threading
from PIL import ImageFont, ImageDraw, Image

warnings.filterwarnings("ignore")

# ================= Console UTF-8 =================
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ================= Dependencies =================
import joblib
import mediapipe as mp

# ================= Paths =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "scraped_data", "svm_sign_model.pkl")

# ================= Config =================
USE_NORMALIZE = True          #  ต้องตรงกับตอนเทรน
USE_EMA_SMOOTH = True
EMA_ALPHA = 0.35

AUTO_PREDICT_DEFAULT = True
WINDOW_SECONDS = 2.0
STEP_SECONDS = 0.35
MIN_HAND_FRAMES = 15
STABLE_FRAMES = 8

CAM_W, CAM_H = 640, 480

# ================= MediaPipe Hands =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================= Feature Config (ตรงกับ TRAIN) =================
HANDS_DIM = 2 * 21 * 3      # 126
FEATURE_OUT_DIM = 252       # mean + std

# ================= Thai Font =================
def load_thai_font(size=18):
    for p in [
        r"C:\Windows\Fonts\THSarabunNew.ttf",
        r"C:\Windows\Fonts\Tahoma.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

THAI_FONT = load_thai_font(18)

def draw_thai_text(frame, text, pos=(10, 10), color=(255, 255, 255)):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=THAI_FONT, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ================= Normalize =================
def _safe(v, eps=1e-6):
    return max(float(v), eps)

def normalize_hand(hand):
    pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
    origin = pts[0]           # wrist
    pts -= origin
    scale = _safe(np.linalg.norm(pts[9]))  # middle MCP
    return (pts / scale).reshape(-1)

# ================= Grabber =================
class LatestFrameGrabber:
    def __init__(self, cam=0):
        self.cap = cv2.VideoCapture(cam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.latest = frame

    def read(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def release(self):
        self.running = False
        self.cap.release()

# ================= Feature Extraction =================
def extract_feature(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        return None, res

    feat = []
    for i in range(2):
        if i < len(res.multi_hand_landmarks):
            h = res.multi_hand_landmarks[i]
            if USE_NORMALIZE:
                feat.extend(normalize_hand(h))
            else:
                for p in h.landmark:
                    feat.extend([p.x, p.y, p.z])
        else:
            feat.extend([0.0] * 63)

    return np.array(feat, dtype=np.float32), res

def make_feature_mean_std(buf):
    arr = np.stack(buf)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)

# ================= Main =================
def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ ไม่พบโมเดล")
        return

    svm = joblib.load(MODEL_PATH)
    grabber = LatestFrameGrabber()

    feats = []
    ema = None
    last_pred = 0
    shown = ""

    print("✅ Q=ออก | A=Auto | S=เริ่ม | E=หยุด")

    while True:
        frame = grabber.read()
        if frame is None:
            continue

        feat, res = extract_feature(frame)

        if feat is not None:
            if USE_EMA_SMOOTH:
                ema = feat if ema is None else EMA_ALPHA * feat + (1 - EMA_ALPHA) * ema
                feat_use = ema
            else:
                feat_use = feat

            feats.append(feat_use)

        if len(feats) > 120:
            feats = feats[-120:]

        now = time.perf_counter()
        if len(feats) >= MIN_HAND_FRAMES and now - last_pred >= STEP_SECONDS:
            X = make_feature_mean_std(feats).reshape(1, -1)
            pred = svm.predict(X)[0]
            conf = np.max(svm.predict_proba(X))
            shown = f"ทำนาย: {pred} ({conf*100:.1f}%)"
            last_pred = now

        if res and res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        if shown:
            frame = draw_thai_text(frame, shown, (10, 10))

        cv2.imshow("Thai Sign SVM (Hands Only)", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    grabber.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
