// ===== Config =====
const API_BASE = window.location.origin;  // same host (FastAPI serves frontend)
const MIN_HAND_FRAMES = 10;
const BUFFER_MAX = 60;
const EMA_ALPHA = 0.35;

// ===== State =====
let isRunning = false;
let frameBuffer = [];
let emaState = null;
let lastPredTime = 0;
let predInterval = 500;
let totalPredictions = 0;
let fpsCount = 0; let fpsLast = performance.now();
let fpsDisplay = 0;
let cameraInstance = null;
let lastPredLabel = "";

// ===== DOM =====
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const predText = document.getElementById("predText");
const confPill = document.getElementById("confPill");
const predMain = document.getElementById("predMain");
const confBarFill = document.getElementById("confBarFill");
const confPct = document.getElementById("confPct");
const top3List = document.getElementById("top3List");
const historyList = document.getElementById("historyList");
const handIndicator = document.getElementById("handIndicator");
const handCount = document.getElementById("handCount");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statFrames = document.getElementById("statFrames");
const statTotal = document.getElementById("statTotal");
const statFPS = document.getElementById("statFPS");
const intervalSlider = document.getElementById("intervalSlider");
const intervalVal = document.getElementById("intervalVal");
const toast = document.getElementById("toast");
const clearHistBtn = document.getElementById("clearHistBtn");

// ===== MediaPipe Hands Setup =====
const hands = new Hands({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

hands.onResults(onHandResults);

// ===== Feature Extraction =====
function extractFeature(multiHandLandmarks) {
  const feat = [];
  for (let i = 0; i < 2; i++) {
    if (i < multiHandLandmarks.length) {
      const lms = multiHandLandmarks[i];
      feat.push(...lms.map(p => [p.x, p.y, p.z]).flat());
    } else {
      feat.push(...new Array(63).fill(0));
    }
  }
  return feat; // length 126
}

function applyEMA(feat) {
  if (!emaState) { emaState = [...feat]; return emaState; }
  emaState = emaState.map((v, i) => EMA_ALPHA * feat[i] + (1 - EMA_ALPHA) * v);
  return emaState;
}

// ===== MediaPipe Results Handler =====
async function onHandResults(results) {
  // FPS counter
  fpsCount++;
  const now = performance.now();
  if (now - fpsLast >= 1000) {
    fpsDisplay = fpsCount;
    fpsCount = 0; fpsLast = now;
    statFPS.textContent = fpsDisplay;
  }

  // Resize overlay canvas to match video
  overlay.width = video.videoWidth || video.clientWidth;
  overlay.height = video.videoHeight || video.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const detected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

  // Update hand indicator
  if (detected) {
    const n = results.multiHandLandmarks.length;
    handIndicator.classList.add("active");
    handCount.textContent = `${n} มือ`;

    // Draw landmarks
    for (const lm of results.multiHandLandmarks) {
      drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: "rgba(0,229,160,0.6)", lineWidth: 1.5 });
      drawLandmarks(ctx, lm, { color: "#00e5a0", lineWidth: 1, radius: 3 });
    }

    // Build feature vector & buffer
    const feat = extractFeature(results.multiHandLandmarks);
    const smoothed = applyEMA(feat);
    frameBuffer.push([...smoothed]);
    if (frameBuffer.length > BUFFER_MAX) frameBuffer.shift();
  } else {
    handIndicator.classList.remove("active");
    handCount.textContent = "ไม่พบมือ";
  }

  statFrames.textContent = frameBuffer.length;

  // Throttled API call
  if (detected && frameBuffer.length >= MIN_HAND_FRAMES && now - lastPredTime >= predInterval) {
    lastPredTime = now;
    await callPredict(results.multiHandLandmarks);
  }
}

// ===== API Call =====
async function callPredict(multiHandLandmarks) {
  const payload = {
    hands: multiHandLandmarks.map(lm => ({
      landmarks: lm.map(p => ({ x: p.x, y: p.y, z: p.z })),
    })),
    buffer: frameBuffer.slice(-60),
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(2000),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateUI(data);
  } catch (e) {
    if (e.name !== "AbortError") {
      console.warn("API error:", e.message);
    }
  }
}

// ===== UI Update =====
function updateUI(data) {
  const { prediction, confidence, top3 } = data;
  totalPredictions++;
  statTotal.textContent = totalPredictions;

  // Main prediction
  predText.textContent = prediction;
  confPill.textContent = `${confidence}%`;
  predMain.textContent = prediction;

  // Flash animation
  predMain.classList.remove("flash");
  void predMain.offsetWidth;
  predMain.classList.add("flash");
  setTimeout(() => predMain.classList.remove("flash"), 400);

  // Confidence bar
  confBarFill.style.width = `${confidence}%`;
  confPct.textContent = `${confidence}%`;

  // Top 3
  top3List.innerHTML = top3.map((item, i) => `
    <div class="top3-item">
      <span class="top3-rank">#${i + 1}</span>
      <span class="top3-label">${item.label}</span>
      <div class="top3-bar-wrap">
        <div class="top3-bar" style="width:${item.confidence}%"></div>
      </div>
      <span class="top3-pct">${item.confidence}%</span>
    </div>
  `).join("");

  // History (deduplicate consecutive same prediction)
  if (prediction !== lastPredLabel || confidence >= 85) {
    lastPredLabel = prediction;
    addHistory(prediction, confidence);
  }
}

function addHistory(label, conf) {
  const empty = historyList.querySelector(".history-empty");
  if (empty) empty.remove();

  const time = new Date().toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const item = document.createElement("div");
  item.className = "history-item";
  item.innerHTML = `
    <span class="history-text">${label}</span>
    <span class="history-meta">
      <span class="history-conf">${conf}%</span>
      <span>${time}</span>
    </span>
  `;
  historyList.insertBefore(item, historyList.firstChild);

  // Keep max 30 items
  while (historyList.children.length > 30) {
    historyList.removeChild(historyList.lastChild);
  }
}

// ===== Camera Control =====
async function startCamera() {
  try {
    cameraInstance = new Camera(video, {
      onFrame: async () => {
        await hands.send({ image: video });
      },
      width: 640,
      height: 480,
    });
    await cameraInstance.start();

    isRunning = true;
    startBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    showToast("✅ เปิดกล้องแล้ว");
    resetBuffer();
  } catch (e) {
    showToast("❌ ไม่สามารถเปิดกล้องได้: " + e.message);
    console.error(e);
  }
}

function stopCamera() {
  if (cameraInstance) {
    cameraInstance.stop();
    cameraInstance = null;
  }
  isRunning = false;
  startBtn.classList.remove("hidden");
  stopBtn.classList.add("hidden");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  showToast("🛑 หยุดกล้องแล้ว");
}

function resetBuffer() {
  frameBuffer = [];
  emaState = null;
  lastPredTime = 0;
}

// ===== Health Check =====
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    if (data.status === "ok" && data.model_loaded) {
      statusDot.className = "dot ok";
      statusText.textContent = "API พร้อมใช้งาน · โมเดลโหลดแล้ว";
    } else {
      statusDot.className = "dot error";
      statusText.textContent = "⚠️ โมเดลยังไม่โหลด";
    }
  } catch {
    statusDot.className = "dot error";
    statusText.textContent = "❌ ไม่สามารถเชื่อมต่อ API";
  }
}

// ===== Toast =====
function showToast(msg, duration = 2500) {
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), duration);
}

// ===== Event Listeners =====
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

clearBtn.addEventListener("click", () => {
  resetBuffer();
  predText.textContent = "— รอสัญญาณมือ —";
  confPill.textContent = "";
  predMain.textContent = "—";
  confBarFill.style.width = "0%";
  confPct.textContent = "0%";
  top3List.innerHTML = '<div class="top3-empty">ยังไม่มีข้อมูล</div>';
  statFrames.textContent = "0";
  showToast("↺ รีเซ็ตบัฟเฟอร์แล้ว");
});

clearHistBtn.addEventListener("click", () => {
  historyList.innerHTML = '<div class="history-empty">ยังไม่มีประวัติ</div>';
  totalPredictions = 0;
  statTotal.textContent = "0";
  lastPredLabel = "";
});

intervalSlider.addEventListener("input", () => {
  predInterval = parseInt(intervalSlider.value);
  intervalVal.textContent = `${predInterval}ms`;
});

// ===== Init =====
checkHealth();
setInterval(checkHealth, 15000);
