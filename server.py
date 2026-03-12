import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime
from collections import deque
import threading
import asyncio
import ollama

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# ============================================================
# Config
# ============================================================

AVAILABLE_MODELS = {
    "qwen3:4b": "Qwen3 4B (Best quality, slower)",
    "qwen3:1.7b": "Qwen3 1.7B (Balanced)",
    "qwen3:0.6b": "Qwen3 0.6B (Fastest, lightweight)",
}

SYSTEM_PROMPT = (
    "You are a caring and supportive academic wellness advisor embedded in an "
    "Ethical AI Surveillance System that monitors student burnout via facial expressions. "
    "Be warm, supportive, and concise (3-5 sentences). "
    "If they seem stressed or burnt out, gently suggest a break or self-care. "
    "If they seem fine, encourage them. Do NOT use emojis."
)


# ============================================================
# Core logic (from main.py / web_app.py)
# ============================================================

def blur_frame(frame):
    return cv2.GaussianBlur(frame, (51, 51), 30)


def calc_burnout_score(emotion_history):
    if not emotion_history:
        return 0.0
    negative = ['sad', 'angry', 'fear', 'disgust']
    scores = []
    for emotions in emotion_history:
        neg = sum(emotions.get(e, 0) for e in negative)
        neu = emotions.get('neutral', 0) * 0.3
        scores.append(min((neg + neu) * 100, 100))
    return float(np.mean(scores))


def check_bias(confidence_history):
    if len(confidence_history) < 5:
        return "Collecting data..."
    avg = np.mean(confidence_history)
    std = np.std(confidence_history)
    if std > 0.25:
        return f"HIGH VARIANCE ({std:.2f}) - potential bias"
    elif avg < 0.4:
        return f"LOW confidence ({avg:.2f}) - check lighting"
    else:
        return f"Stable ({avg:.2f} +/-{std:.2f})"


def generate_report(emotion_history, burnout_scores, session_start):
    now = datetime.now()
    duration = (now - session_start).seconds
    lines = []
    lines.append("=" * 55)
    lines.append("  ETHICAL AI SURVEILLANCE SYSTEM - SESSION REPORT")
    lines.append("  Smart Academic Burnout Predictor")
    lines.append("  Team: Project Protocol")
    lines.append("=" * 55)
    lines.append(f"  Session Start : {session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Session End   : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Duration      : {duration // 60}m {duration % 60}s")
    lines.append(f"  Samples       : {len(emotion_history)}")
    lines.append("-" * 55)

    if emotion_history:
        avg_emotions = {}
        for emotions in emotion_history:
            for k, v in emotions.items():
                avg_emotions[k] = avg_emotions.get(k, 0) + v
        for k in avg_emotions:
            avg_emotions[k] /= len(emotion_history)
        lines.append("  AVERAGE EMOTION BREAKDOWN:")
        for emotion, val in sorted(avg_emotions.items(), key=lambda x: -x[1]):
            bar = "#" * int(val * 30)
            lines.append(f"    {emotion:<10} {val:>6.1%}  {bar}")

    lines.append("-" * 55)
    if burnout_scores:
        avg_b = np.mean(burnout_scores)
        max_b = np.max(burnout_scores)
        lines.append(f"  AVG BURNOUT SCORE : {avg_b:.1f}%")
        lines.append(f"  PEAK BURNOUT      : {max_b:.1f}%")
        if avg_b < 25:
            lines.append("  STATUS: LOW RISK - Student appears engaged")
        elif avg_b < 50:
            lines.append("  STATUS: MODERATE - Monitor for signs of fatigue")
        elif avg_b < 75:
            lines.append("  STATUS: HIGH RISK - Recommend break / support")
        else:
            lines.append("  STATUS: CRITICAL - Immediate intervention suggested")
    lines.append("-" * 55)
    lines.append("  PRIVACY NOTE: No facial data was stored.")
    lines.append("  All processing was done locally in real-time.")
    lines.append("  Raw frames were blurred and never saved.")
    lines.append("=" * 55)
    return "\n".join(lines)


def draw_overlays(display, session):
    h, w = display.shape[:2]
    sidebar_w = 250
    overlay = display.copy()
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    cv2.putText(display, "ETHICAL AI SURVEILLANCE", (w - sidebar_w + 8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    cv2.putText(display, "Burnout Predictor", (w - sidebar_w + 8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    emotions = session.last_emotions
    colors = {
        'happy': (0, 200, 0), 'sad': (200, 100, 0), 'angry': (0, 0, 220),
        'surprise': (0, 220, 220), 'fear': (180, 0, 180),
        'disgust': (0, 140, 140), 'neutral': (180, 180, 180),
    }
    if emotions:
        for i, (em, score) in enumerate(sorted(emotions.items(), key=lambda x: -x[1])):
            y = 65 + i * 20
            color = colors.get(em, (200, 200, 200))
            cv2.rectangle(display, (w - sidebar_w + 8, y), (w - sidebar_w + 108, y + 14), (40, 40, 40), -1)
            cv2.rectangle(display, (w - sidebar_w + 8, y), (w - sidebar_w + 8 + int(100 * score), y + 14), color, -1)
            cv2.putText(display, f"{em}: {score:.0%}", (w - sidebar_w + 115, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    if session.burnout_scores:
        bs = session.burnout_scores[-1]
        if bs < 25:
            level, color = "LOW", (0, 200, 0)
        elif bs < 50:
            level, color = "MODERATE", (0, 200, 255)
        elif bs < 75:
            level, color = "HIGH", (0, 100, 255)
        else:
            level, color = "CRITICAL", (0, 0, 255)
        y = 220
        cv2.putText(display, "BURNOUT RISK", (w - sidebar_w + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(display, (w - sidebar_w + 8, y + 8), (w - sidebar_w + 168, y + 26), (40, 40, 40), -1)
        cv2.rectangle(display, (w - sidebar_w + 8, y + 8),
                      (w - sidebar_w + 8 + int(160 * bs / 100), y + 26), color, -1)
        cv2.putText(display, f"{bs:.0f}% {level}", (w - sidebar_w + 12, y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    scale_x, scale_y = w / 320, h / 240
    for res in session.last_results:
        bx, by, bw, bh = res['box']
        bx, by = max(int(bx * scale_x), 0), max(int(by * scale_y), 0)
        bw, bh = min(int(bw * scale_x), w - bx), min(int(bh * scale_y), h - by)
        top_em = max(res['emotions'], key=res['emotions'].get)
        c = (0, 255, 0) if top_em == 'happy' else (0, 165, 255) if top_em == 'neutral' else (0, 0, 255)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), c, 2)
        cv2.putText(display, f"{top_em} ({res['emotions'][top_em]:.0%})",
                    (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    return display


def send_to_ollama(messages, model):
    try:
        response = ollama.chat(model=model, messages=messages, think=True)
        return response.message.thinking, response.message.content
    except Exception as e:
        return None, f"Ollama error: {e}"


# ============================================================
# Session Manager
# ============================================================

class SessionManager:
    def __init__(self):
        self.cap = None
        self.emotion_history = deque(maxlen=300)
        self.burnout_scores = deque(maxlen=300)
        self.confidence_history = deque(maxlen=100)
        self.session_start = None
        self.privacy_mode = True
        self.last_emotions = {}
        self.last_results = []
        self.model = "qwen3:4b"
        self.running = False
        self.chat_history = []
        self.lock = threading.Lock()
        self._analyzing = False

    def start(self, model=None):
        if self.running:
            return False
        if model and model in AVAILABLE_MODELS:
            self.model = model
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        self.emotion_history.clear()
        self.burnout_scores.clear()
        self.confidence_history.clear()
        self.session_start = datetime.now()
        self.last_emotions = {}
        self.last_results = []
        self.chat_history = []
        self.running = True
        self._analyzing = False
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def analyze_frame(self, frame):
        try:
            small = cv2.resize(frame, (320, 240))
            results = DeepFace.analyze(
                small, actions=['emotion'], enforce_detection=False,
                silent=True, detector_backend='opencv'
            )
            if not isinstance(results, list):
                results = [results]
            parsed = []
            for r in results:
                region = r.get('region', {})
                raw = r.get('emotion', {})
                total = sum(raw.values()) if raw else 1
                emotions = {k.lower(): v / total for k, v in raw.items()} if total else {}
                parsed.append({
                    'box': [region.get('x', 0), region.get('y', 0),
                            region.get('w', 0), region.get('h', 0)],
                    'emotions': emotions,
                })
            if parsed:
                with self.lock:
                    self.last_results = parsed
                    self.last_emotions = parsed[0]['emotions']
                    self.emotion_history.append(self.last_emotions)
                    top = max(self.last_emotions, key=self.last_emotions.get)
                    self.confidence_history.append(self.last_emotions[top])
                    burnout = calc_burnout_score(list(self.emotion_history)[-30:])
                    self.burnout_scores.append(burnout)
        except Exception:
            pass
        finally:
            self._analyzing = False

    def get_state(self):
        with self.lock:
            elapsed = 0
            if self.session_start:
                elapsed = (datetime.now() - self.session_start).seconds
            return {
                "emotions": dict(self.last_emotions),
                "burnout": float(self.burnout_scores[-1]) if self.burnout_scores else 0.0,
                "bias": check_bias(list(self.confidence_history)),
                "samples": len(self.emotion_history),
                "elapsed": elapsed,
                "privacy": self.privacy_mode,
                "running": self.running,
                "model": self.model,
            }


session = SessionManager()

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="Burnout Predictor API")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Request models ---

class SessionStartRequest(BaseModel):
    model: Optional[str] = None

class SettingsRequest(BaseModel):
    privacy_mode: Optional[bool] = None
    model: Optional[str] = None

class ChatRequest(BaseModel):
    message: str


# --- Routes ---

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/models")
async def get_models():
    return {
        "models": [
            {"id": k, "name": v} for k, v in AVAILABLE_MODELS.items()
        ],
        "current": session.model,
    }


@app.post("/api/session/start")
async def start_session(req: SessionStartRequest = SessionStartRequest()):
    if session.running:
        return JSONResponse({"error": "Session already running"}, status_code=400)
    ok = session.start(req.model)
    if not ok:
        return JSONResponse({"error": "Cannot open webcam"}, status_code=500)
    return {"status": "started", "model": session.model}


@app.post("/api/session/stop")
async def stop_session():
    if not session.running:
        return JSONResponse({"error": "No session running"}, status_code=400)
    report = generate_report(
        list(session.emotion_history),
        list(session.burnout_scores),
        session.session_start,
    )
    session.stop()
    return {"status": "stopped", "report": report}


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    if req.privacy_mode is not None:
        session.privacy_mode = req.privacy_mode
    if req.model is not None and req.model in AVAILABLE_MODELS:
        session.model = req.model
    return {"ok": True}


@app.get("/api/video_feed")
async def video_feed():
    async def generate_frames():
        while session.running and session.cap and session.cap.isOpened():
            ret, frame = session.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            if session.privacy_mode:
                display = blur_frame(display)

            if not session._analyzing:
                session._analyzing = True
                t = threading.Thread(target=session.analyze_frame, args=(frame,), daemon=True)
                t.start()

            display = draw_overlays(display, session)

            _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )


@app.websocket("/ws/emotions")
async def websocket_emotions(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            state = session.get_state()
            await ws.send_json(state)
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass


@app.post("/api/report")
async def get_report():
    if not session.session_start:
        return JSONResponse({"error": "No session data"}, status_code=400)
    report = generate_report(
        list(session.emotion_history),
        list(session.burnout_scores),
        session.session_start,
    )
    return {"report": report}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    report = ""
    if session.session_start:
        report = generate_report(
            list(session.emotion_history),
            list(session.burnout_scores),
            session.session_start,
        )
    content = f"{req.message}\n\n[Current session context]:\n{report}" if report else req.message
    session.chat_history.append({"role": "user", "content": content})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.chat_history[-20:]
    loop = asyncio.get_event_loop()
    thinking, answer = await loop.run_in_executor(None, send_to_ollama, messages, session.model)
    session.chat_history.append({"role": "assistant", "content": answer})
    result = {"response": answer}
    if thinking:
        result["thinking"] = thinking
    return result


@app.post("/api/support")
async def support():
    if not session.session_start:
        return JSONResponse({"error": "No session data"}, status_code=400)
    report = generate_report(
        list(session.emotion_history),
        list(session.burnout_scores),
        session.session_start,
    )
    session.chat_history.append({
        "role": "user",
        "content": f"Here is the student's burnout report. Give supportive feedback:\n\n{report}",
    })
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.chat_history[-20:]
    loop = asyncio.get_event_loop()
    thinking, answer = await loop.run_in_executor(None, send_to_ollama, messages, session.model)
    session.chat_history.append({"role": "assistant", "content": answer})
    result = {"report": report, "response": answer}
    if thinking:
        result["thinking"] = thinking
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="localhost", port=8000)
