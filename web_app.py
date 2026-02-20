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
import time
import ollama
import requests
import base64
import streamlit as st

OLLAMA_MODEL = "qwen3:4b"
AUTO_REPORT_INTERVAL = 45 * 60
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

SYSTEM_PROMPT = (
    "You are a caring and supportive academic wellness advisor embedded in an "
    "Ethical AI Surveillance System that monitors student burnout via facial expressions. "
    "Be warm, supportive, and concise (3-5 sentences). "
    "If they seem stressed or burnt out, gently suggest a break or self-care. "
    "If they seem fine, encourage them. Do NOT use emojis."
)


# ============================================================
# Reused logic from main.py
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
    return np.mean(scores)


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
    lines.append("ETHICAL AI SURVEILLANCE SYSTEM - SESSION REPORT")
    lines.append("Smart Academic Burnout Predictor | Team: Project Protocol")
    lines.append(f"Session: {session_start.strftime('%H:%M:%S')} - {now.strftime('%H:%M:%S')} ({duration // 60}m {duration % 60}s)")
    lines.append(f"Samples: {len(emotion_history)}")

    if emotion_history:
        avg_emotions = {}
        for emotions in emotion_history:
            for k, v in emotions.items():
                avg_emotions[k] = avg_emotions.get(k, 0) + v
        for k in avg_emotions:
            avg_emotions[k] /= len(emotion_history)
        lines.append("\nEMOTION BREAKDOWN:")
        for emotion, val in sorted(avg_emotions.items(), key=lambda x: -x[1]):
            lines.append(f"  {emotion:<10} {val:>6.1%}")

    if burnout_scores:
        avg_b = np.mean(burnout_scores)
        max_b = np.max(burnout_scores)
        lines.append(f"\nAVG BURNOUT: {avg_b:.1f}% | PEAK: {max_b:.1f}%")
        if avg_b < 25:
            lines.append("STATUS: LOW RISK")
        elif avg_b < 50:
            lines.append("STATUS: MODERATE")
        elif avg_b < 75:
            lines.append("STATUS: HIGH RISK")
        else:
            lines.append("STATUS: CRITICAL")

    lines.append("\nPRIVACY: No facial data stored. All processing local.")
    return "\n".join(lines)


def send_to_ollama(messages):
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages, think=True)
        return response.message.thinking, response.message.content
    except Exception as e:
        return None, f"Ollama error: {e}"


def get_tts_audio(text):
    if not SARVAM_API_KEY:
        return None
    try:
        resp = requests.post(
            SARVAM_TTS_URL,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "content-type": "application/json",
            },
            json={
                "text": text[:2500],
                "target_language_code": "en-IN",
                "speaker": "priya",
                "pace": 1.1,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": "bulbul:v3",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return base64.b64decode(resp.json()["audios"][0])
    except Exception:
        return None


# ============================================================
# Session state init
# ============================================================

def init_state():
    defaults = {
        "emotion_history": deque(maxlen=300),
        "burnout_scores": deque(maxlen=300),
        "confidence_history": deque(maxlen=100),
        "session_start": datetime.now(),
        "privacy_mode": True,
        "last_emotions": {},
        "last_results": [],
        "analyzing": False,
        "camera_running": True,
        "last_auto_report": time.time(),
        "chat_messages": [],
        "ollama_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# Detection
# ============================================================

def analyze_frame(frame):
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
            parsed.append({'box': [region.get('x', 0), region.get('y', 0),
                                   region.get('w', 0), region.get('h', 0)],
                           'emotions': emotions})
        if parsed:
            st.session_state.last_results = parsed
            st.session_state.last_emotions = parsed[0]['emotions']
            st.session_state.emotion_history.append(st.session_state.last_emotions)
            top = max(st.session_state.last_emotions, key=st.session_state.last_emotions.get)
            st.session_state.confidence_history.append(st.session_state.last_emotions[top])
            burnout = calc_burnout_score(list(st.session_state.emotion_history)[-30:])
            st.session_state.burnout_scores.append(burnout)
    except Exception:
        pass
    finally:
        st.session_state.analyzing = False


def draw_overlays(display):
    h, w = display.shape[:2]
    sidebar_w = 250
    overlay = display.copy()
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    cv2.putText(display, "ETHICAL AI SURVEILLANCE", (w - sidebar_w + 8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    cv2.putText(display, "Burnout Predictor", (w - sidebar_w + 8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Emotion bars
    emotions = st.session_state.last_emotions
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

    # Burnout gauge
    if st.session_state.burnout_scores:
        bs = st.session_state.burnout_scores[-1]
        if bs < 25:
            level, color = "LOW", (0, 200, 0)
        elif bs < 50:
            level, color = "MODERATE", (0, 200, 255)
        elif bs < 75:
            level, color = "HIGH", (0, 100, 255)
        else:
            level, color = "CRITICAL", (0, 0, 255)
        y = 220
        cv2.putText(display, "BURNOUT RISK", (w - sidebar_w + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(display, (w - sidebar_w + 8, y + 8), (w - sidebar_w + 168, y + 26), (40, 40, 40), -1)
        cv2.rectangle(display, (w - sidebar_w + 8, y + 8), (w - sidebar_w + 8 + int(160 * bs / 100), y + 26), color, -1)
        cv2.putText(display, f"{bs:.0f}% {level}", (w - sidebar_w + 12, y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # Face boxes
    scale_x, scale_y = w / 320, h / 240
    for res in st.session_state.last_results:
        bx, by, bw, bh = res['box']
        bx, by = max(int(bx * scale_x), 0), max(int(by * scale_y), 0)
        bw, bh = min(int(bw * scale_x), w - bx), min(int(bh * scale_y), h - by)
        top_em = max(res['emotions'], key=res['emotions'].get)
        c = (0, 255, 0) if top_em == 'happy' else (0, 165, 255) if top_em == 'neutral' else (0, 0, 255)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), c, 2)
        cv2.putText(display, f"{top_em} ({res['emotions'][top_em]:.0%})",
                    (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    return display


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Ethical AI Surveillance - Burnout Predictor", layout="wide")
init_state()

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }
    h1 { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("Ethical AI Surveillance System - Smart Academic Burnout Predictor")
st.caption("Team: Project Protocol | Privacy-first emotion monitoring")

left_col, right_col = st.columns([3, 2])

# ---- LEFT: Camera ----
with left_col:
    st.subheader("Live Camera Feed")
    cam_controls = st.columns(3)
    with cam_controls[0]:
        privacy = st.toggle("Privacy Mode (Blur)", value=st.session_state.privacy_mode)
        st.session_state.privacy_mode = privacy
    with cam_controls[1]:
        run_camera = st.toggle("Camera On", value=True)
    with cam_controls[2]:
        elapsed = (datetime.now() - st.session_state.session_start).seconds
        st.metric("Session", f"{elapsed // 60:02d}:{elapsed % 60:02d}")

    camera_placeholder = st.empty()

    # Stats row
    stats_cols = st.columns(4)
    with stats_cols[0]:
        samples_ph = st.empty()
    with stats_cols[1]:
        burnout_ph = st.empty()
    with stats_cols[2]:
        top_emotion_ph = st.empty()
    with stats_cols[3]:
        bias_ph = st.empty()

    # Emotion chart
    chart_placeholder = st.empty()

# ---- RIGHT: Chat ----
with right_col:
    st.subheader("AI Wellness Advisor")

    # Report + AI support button
    btn_cols = st.columns(2)
    with btn_cols[0]:
        report_btn = st.button("Generate Report + AI Support", type="primary", use_container_width=True)
    with btn_cols[1]:
        clear_btn = st.button("Clear Chat", use_container_width=True)

    if clear_btn:
        st.session_state.chat_messages = []
        st.session_state.ollama_history = []
        st.rerun()

    # Chat display
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.chat_messages:
            st.info("Hey! I'm your AI wellness advisor. Press the button above to get a burnout report, or just chat with me below.")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["display"])
                if "audio" in msg and msg["audio"]:
                    st.audio(msg["audio"], format="audio/wav")

    # Chat input
    user_input = st.chat_input("Chat with your wellness advisor...")


# ============================================================
# Handle report button
# ============================================================

if report_btn:
    report = generate_report(
        list(st.session_state.emotion_history),
        list(st.session_state.burnout_scores),
        st.session_state.session_start,
    )
    # Save report
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(report)

    st.session_state.chat_messages.append({
        "role": "assistant", "display": f"**Session Report:**\n```\n{report}\n```"
    })

    # Send to LLM
    st.session_state.ollama_history.append({
        "role": "user",
        "content": f"Here is the student's burnout report. Give supportive feedback:\n\n{report}"
    })
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.ollama_history[-20:]

    with st.spinner("AI advisor is thinking..."):
        thinking, answer = send_to_ollama(messages)

    st.session_state.ollama_history.append({"role": "assistant", "content": answer})

    # TTS
    audio_bytes = get_tts_audio(answer)

    display_text = answer
    if thinking:
        display_text = f"*Thinking: {thinking[:200]}...*\n\n{answer}" if len(thinking) > 200 else f"*Thinking: {thinking}*\n\n{answer}"

    st.session_state.chat_messages.append({
        "role": "assistant", "display": display_text, "audio": audio_bytes
    })

    # Save LLM response to report file
    with open(fname, 'a', encoding='utf-8') as f:
        if thinking:
            f.write(f"\n\nAI REASONING:\n{thinking}")
        f.write(f"\n\nAI WELLNESS ADVISOR:\n{answer}")

    st.session_state.last_auto_report = time.time()
    st.rerun()


# ============================================================
# Handle chat input
# ============================================================

if user_input:
    st.session_state.chat_messages.append({"role": "user", "display": user_input})

    # Add current session context
    report = generate_report(
        list(st.session_state.emotion_history),
        list(st.session_state.burnout_scores),
        st.session_state.session_start,
    )
    content = f"{user_input}\n\n[Current session context]:\n{report}"
    st.session_state.ollama_history.append({"role": "user", "content": content})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.ollama_history[-20:]

    with st.spinner("AI advisor is thinking..."):
        thinking, answer = send_to_ollama(messages)

    st.session_state.ollama_history.append({"role": "assistant", "content": answer})

    audio_bytes = get_tts_audio(answer)

    display_text = answer
    if thinking:
        display_text = f"*Thinking: {thinking[:200]}...*\n\n{answer}" if len(thinking) > 200 else f"*Thinking: {thinking}*\n\n{answer}"

    st.session_state.chat_messages.append({
        "role": "assistant", "display": display_text, "audio": audio_bytes
    })
    st.rerun()


# ============================================================
# Auto report check (every 45 min)
# ============================================================

if time.time() - st.session_state.last_auto_report >= AUTO_REPORT_INTERVAL:
    if len(st.session_state.emotion_history) > 0:
        st.session_state.last_auto_report = time.time()
        # Trigger same as report button
        report = generate_report(
            list(st.session_state.emotion_history),
            list(st.session_state.burnout_scores),
            st.session_state.session_start,
        )
        st.session_state.ollama_history.append({
            "role": "user",
            "content": f"[AUTO 45-MIN REPORT]\n{report}\n\nPlease give supportive feedback."
        })
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.ollama_history[-20:]
        thinking, answer = send_to_ollama(messages)
        st.session_state.ollama_history.append({"role": "assistant", "content": answer})
        audio_bytes = get_tts_audio(answer)
        st.session_state.chat_messages.append({
            "role": "assistant",
            "display": f"**[Auto 45-min Report]**\n```\n{report}\n```\n\n{answer}",
            "audio": audio_bytes,
        })
        st.rerun()


# ============================================================
# Camera loop (runs each streamlit rerun)
# ============================================================

if run_camera:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        frame_count = 0
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            if st.session_state.privacy_mode:
                display = blur_frame(display)

            # Run detection every 5 frames
            if frame_count % 5 == 0 and not st.session_state.analyzing:
                st.session_state.analyzing = True
                analyze_frame(frame)

            display = draw_overlays(display)

            # Convert BGR to RGB for streamlit
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            # Update stats
            samples_ph.metric("Samples", len(st.session_state.emotion_history))
            if st.session_state.burnout_scores:
                burnout_ph.metric("Burnout", f"{st.session_state.burnout_scores[-1]:.0f}%")
            if st.session_state.last_emotions:
                top = max(st.session_state.last_emotions, key=st.session_state.last_emotions.get)
                top_emotion_ph.metric("Top Emotion", f"{top} ({st.session_state.last_emotions[top]:.0%})")
            bias_ph.metric("Bias Check", check_bias(list(st.session_state.confidence_history)))

            frame_count += 1
            time.sleep(0.03)  # ~30fps cap

        cap.release()
    else:
        camera_placeholder.error("Cannot open webcam!")
else:
    camera_placeholder.info("Camera is off. Toggle 'Camera On' to start.")
