# 🤟 Thai Sign Language Web App

แปลภาษามือไทย Real-time ผ่าน Browser โดยใช้ MediaPipe + SVM

## 📁 โครงสร้างโปรเจกต์

```
thai-sign-web/
├── backend/
│   ├── main.py                 ← FastAPI server + prediction logic
│   ├── requirements.txt        ← Python dependencies
│   ├── Dockerfile              ← Docker image
│   └── svm_sign_model.pkl      ← ← วางไฟล์โมเดลตรงนี้ !!
├── frontend/
│   ├── index.html              ← หน้าเว็บหลัก
│   └── static/
│       ├── css/style.css       ← สไตล์ชีต
│       └── js/app.js           ← MediaPipe + API logic
├── docker-compose.yml
├── nginx.conf
└── README.md
```

---

## 🚀 วิธีรัน (เลือกวิธีที่ต้องการ)

---

### วิธีที่ 1 — รันตรง (แนะนำสำหรับ Development)

#### ขั้นที่ 1: วางไฟล์โมเดล
```bash
cp svm_sign_model.pkl thai-sign-web/backend/svm_sign_model.pkl
```

#### ขั้นที่ 2: ติดตั้ง Python dependencies
```bash
cd thai-sign-web/backend
pip install -r requirements.txt
```

#### ขั้นที่ 3: รัน FastAPI server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### ขั้นที่ 4: เปิด Browser
```
http://localhost:8000
```

✅ FastAPI จะ serve ทั้ง API และ Frontend จาก port เดียวกัน

---

### วิธีที่ 2 — Docker Compose (สำหรับ Production)

#### ขั้นที่ 1: วางไฟล์โมเดล
```bash
cp svm_sign_model.pkl thai-sign-web/backend/svm_sign_model.pkl
```

#### ขั้นที่ 2: Build และรัน
```bash
cd thai-sign-web
docker-compose up --build
```

#### ขั้นที่ 3: เปิด Browser
```
http://localhost
```

---

## 🌐 Deploy บน Server จริง (VPS/Cloud)

### ต้องการ HTTPS สำหรับ Webcam Access

Browser จะอนุญาต getUserMedia เฉพาะบน:
- `localhost` (local dev)
- HTTPS domains เท่านั้น

#### ตัวเลือก Deploy:
1. **Render.com** — Deploy FastAPI ฟรี, รองรับ HTTPS อัตโนมัติ
2. **Railway.app** — รองรับ Dockerfile deploy
3. **DigitalOcean / AWS** + Let's Encrypt SSL

#### Render.com (ง่ายที่สุด):
1. Push โปรเจกต์ขึ้น GitHub
2. สร้าง Web Service ใหม่ → เลือก repo
3. Build Command: `pip install -r backend/requirements.txt`
4. Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Environment Variable: ไม่ต้องตั้งเพิ่มเติม
6. อัปโหลดไฟล์ `svm_sign_model.pkl` ผ่าน Render Disk หรือ include ใน repo

---

## ⚙️ API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | ตรวจสอบสถานะ API + โมเดล |
| POST | `/predict` | รับ Hand Landmarks → คืนผลทำนาย |
| GET | `/` | หน้าเว็บ Frontend |

### POST /predict — Request Body
```json
{
  "hands": [
    {
      "landmarks": [
        {"x": 0.5, "y": 0.3, "z": 0.01},
        ...
      ]
    }
  ],
  "buffer": [[...feature vectors...]]
}
```

### POST /predict — Response
```json
{
  "prediction": "ตื่นนอนตอนเช้า",
  "confidence": 87.3,
  "top3": [
    {"label": "ตื่นนอนตอนเช้า", "confidence": 87.3},
    {"label": "ล้างหน้าแปรงฟัน", "confidence": 8.1},
    {"label": "อาบน้ำ", "confidence": 4.6}
  ],
  "feature_dim": 252
}
```

---

## 🔧 ปรับแต่งค่า

### Backend (`backend/main.py`)
```python
USE_NORMALIZE = True   # ต้องตรงกับตอนเทรนโมเดล
```

### Frontend (`frontend/static/js/app.js`)
```javascript
const MIN_HAND_FRAMES = 10;  // จำนวนเฟรมขั้นต่ำก่อน predict
const BUFFER_MAX = 60;        // ขนาด buffer สูงสุด (เฟรม)
const EMA_ALPHA = 0.35;       // ค่า smoothing (0.1=ช้า, 0.9=เร็ว)
```

ปุ่ม **"ความเร็ว API"** บนหน้าเว็บ = ช่วงเวลาระหว่าง API calls (200-1500ms)

---

## 🐛 แก้ปัญหาที่พบบ่อย

| ปัญหา | วิธีแก้ |
|-------|---------|
| กล้องไม่เปิด | ตรวจสอบว่าเปิดบน localhost หรือ HTTPS |
| API ไม่ตอบ | ตรวจสอบว่ารัน uvicorn แล้ว และ port 8000 ไม่ถูกบล็อก |
| โมเดลไม่โหลด | ตรวจสอบว่าวางไฟล์ `.pkl` ใน `backend/` |
| ทำนายผิดทุกครั้ง | ตรวจสอบ `USE_NORMALIZE` ต้องตรงกับตอนเทรน |
| MediaPipe ไม่โหลด | ตรวจสอบ Internet connection (CDN scripts) |

---

## 📊 ฟีเจอร์ทั้งหมดบนหน้าเว็บ

- ✅ แสดงภาพกล้อง Real-time (Mirror)
- ✅ วาด Hand Landmarks บนภาพ
- ✅ แสดงผลทำนายบน Video overlay
- ✅ แสดง Top 3 ความน่าจะเป็น
- ✅ แสดง Confidence Bar
- ✅ ประวัติการทำนาย
- ✅ ปรับความเร็ว API call
- ✅ แสดง FPS กล้อง
- ✅ Health check status
- ✅ รีเซ็ตบัฟเฟอร์ได้
