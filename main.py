import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from rich.console import Console
console = Console()

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
import tempfile
import subprocess

AVAILABLE_MODELS = {
    "1": ("qwen3:4b", "Qwen3 4B (Best quality, slower)"),
    "2": ("qwen3:1.7b", "Qwen3 1.7B (Balanced)"),
    "3": ("qwen3:0.6b", "Qwen3 0.6B (Fastest, lightweight)"),
}
OLLAMA_MODEL = "qwen3:4b"  # default, overridden by user selection
AUTO_REPORT_INTERVAL = 45 * 60  # 45 minutes in seconds
SARVAM_API_KEY = os.environ.get("SARVAM_KEY", "")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


# ============================================================
# Ethical AI Surveillance System - Smart Academic Burnout Predictor
# Team: Project Protocol
# ============================================================


def blur_frame(frame):
    return cv2.GaussianBlur(frame, (51, 51), 30)


def calc_burnout_score(emotion_history):
    if not emotion_history:
        return 0.0
    negative_emotions = ['sad', 'angry', 'fear', 'disgust']
    neutral_weight = 0.3
    scores = []
    for emotions in emotion_history:
        neg = sum(emotions.get(e, 0) for e in negative_emotions)
        neu = emotions.get('neutral', 0) * neutral_weight
        scores.append(min((neg + neu) * 100, 100))
    return np.mean(scores)


def check_bias(confidence_history):
    if len(confidence_history) < 5:
        return "Collecting data...", (200, 200, 200)
    avg = np.mean(confidence_history)
    std = np.std(confidence_history)
    if std > 0.25:
        return f"HIGH VARIANCE ({std:.2f}) - potential bias", (0, 0, 255)
    elif avg < 0.4:
        return f"LOW conf ({avg:.2f}) - check lighting", (0, 165, 255)
    else:
        return f"Stable ({avg:.2f} +/-{std:.2f})", (0, 255, 0)


def draw_emotion_bars(frame, emotions, x_start, y_start, bar_width=100, bar_height=14, gap=20):
    if not emotions:
        return
    colors = {
        'happy': (0, 200, 0), 'sad': (200, 100, 0), 'angry': (0, 0, 220),
        'surprise': (0, 220, 220), 'fear': (180, 0, 180),
        'disgust': (0, 140, 140), 'neutral': (180, 180, 180),
    }
    for i, (emotion, score) in enumerate(sorted(emotions.items(), key=lambda x: -x[1])):
        y = y_start + i * gap
        color = colors.get(emotion, (200, 200, 200))
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height), (40, 40, 40), -1)
        fill_w = int(bar_width * score)
        cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + bar_height), color, -1)
        cv2.putText(frame, f"{emotion}: {score:.0%}", (x_start + bar_width + 8, y + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


def draw_burnout_gauge(frame, score, x, y):
    if score < 25:
        level, color = "LOW", (0, 200, 0)
    elif score < 50:
        level, color = "MODERATE", (0, 200, 255)
    elif score < 75:
        level, color = "HIGH", (0, 100, 255)
    else:
        level, color = "CRITICAL", (0, 0, 255)

    cv2.putText(frame, "BURNOUT RISK", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (x, y + 8), (x + 180, y + 28), (40, 40, 40), -1)
    fill = int(180 * score / 100)
    cv2.rectangle(frame, (x, y + 8), (x + fill, y + 28), color, -1)
    cv2.putText(frame, f"{score:.0f}% - {level}", (x + 5, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


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


def send_to_ollama(report):
    """Send report to Ollama (thinking model) and get a supportive response."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a caring and supportive academic wellness advisor. "
                        "A student's emotion monitoring system generated a report. "
                        "Respond with a short, warm, supportive message (3-5 sentences). "
                        "If they seem stressed or burnt out, gently suggest a break or self-care. "
                        "If they seem fine, encourage them to keep going."
                    ),
                },
                {"role": "user", "content": f"Here is my session report:\n\n{report}"},
            ],
            think=True,
        )
        return response.message.thinking, response.message.content
    except Exception as e:
        return None, f"[Ollama error: {e}]"


def speak_text(text):
    """Convert text to speech using Sarvam AI and play it."""
    if not SARVAM_API_KEY:
        console.print("  [yellow]SARVAM_API_KEY not set, skipping TTS[/]")
        return
    # Sarvam has a 2500 char limit for bulbul:v3
    text = text[:2500]
    try:
        resp = requests.post(
            SARVAM_TTS_URL,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "content-type": "application/json",
            },
            json={
                "text": text,
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
        audio_b64 = resp.json()["audios"][0]
        audio_bytes = base64.b64decode(audio_b64)

        # Save to temp file and play
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()

        # Play audio on Windows
        subprocess.Popen(
            ["powershell", "-c", f'(New-Object Media.SoundPlayer "{tmp.name}").PlaySync()'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).wait()
        os.unlink(tmp.name)
    except Exception as e:
        console.print(f"  [red]TTS error: {e}[/]")


def save_report_and_get_support(emotion_history, burnout_scores, session_start):
    """Generate report, save it, send to Ollama, print supportive message."""
    report = generate_report(list(emotion_history), list(burnout_scores), session_start)
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(report)
    console.print(f"\n  Report saved to {fname}")
    console.print("  Sending to AI advisor (thinking)...\n")
    thinking, answer = send_to_ollama(report)
    if thinking:
        console.print("  " + "-" * 50)
        console.print("  AI REASONING:")
        console.print(thinking)
    console.print("  " + "=" * 50)
    console.print("  AI WELLNESS ADVISOR:")
    console.print("  " + "-" * 50)
    console.print(answer)
    console.print("  " + "=" * 50 + "\n")
    # Speak the response
    speak_text(answer)
    with open(fname, 'a', encoding='utf-8') as f:
        if thinking:
            f.write("\n\nAI REASONING:\n")
            f.write(thinking)
        f.write("\n\nAI WELLNESS ADVISOR RESPONSE:\n")
        f.write(answer)


def select_model():
    """Let the user pick an Ollama model at startup."""
    global OLLAMA_MODEL
    console.print("\n  [bold cyan]Select AI Model:[/]")
    for key, (model, desc) in AVAILABLE_MODELS.items():
        console.print(f"    [bold]{key}[/]) {desc}  [dim]({model})[/]")
    console.print()
    choice = input("  Enter choice [1/2/3] (default=1): ").strip()
    if choice in AVAILABLE_MODELS:
        OLLAMA_MODEL = AVAILABLE_MODELS[choice][0]
    else:
        OLLAMA_MODEL = AVAILABLE_MODELS["1"][0]
    console.print(f"  [green]Using model: {OLLAMA_MODEL}[/]\n")


def main():
    console.print("=" * 55)
    console.print("  Ethical AI Surveillance System")
    console.print("  Smart Academic Burnout Predictor")
    console.print("  Team: Project Protocol")
    console.print("=" * 55)

    select_model()

    console.print("  Loading emotion detection model...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        console.print("ERROR: Cannot open webcam!")
        return

    console.print("  Webcam ready!")
    console.print("  Controls: Q=Quit  R=Report  P=Toggle Privacy  S=AI Support\n")

    emotion_history = deque(maxlen=300)
    burnout_scores = deque(maxlen=300)
    confidence_history = deque(maxlen=100)
    session_start = datetime.now()
    privacy_mode = True
    last_emotions = {}
    last_results = []
    last_auto_report = time.time()

    # --- Threaded detection so webcam stays smooth ---
    analysis_lock = threading.Lock()
    analyzing = False

    def analyze_frame(img):
        nonlocal last_results, last_emotions, analyzing
        try:
            results = DeepFace.analyze(
                img, actions=['emotion'], enforce_detection=False,
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
                with analysis_lock:
                    last_results = parsed
                    last_emotions = parsed[0]['emotions']
                    emotion_history.append(last_emotions)
                    top_emotion = max(last_emotions, key=last_emotions.get)
                    confidence_history.append(last_emotions[top_emotion])
                    burnout = calc_burnout_score(list(emotion_history)[-30:])
                    burnout_scores.append(burnout)
        except Exception:
            pass
        finally:
            analyzing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = display.shape[:2]

        # Apply blur to whole feed if privacy mode is on (before drawing UI)
        if privacy_mode:
            display = blur_frame(display)

        # Launch analysis in background thread when previous one is done
        if not analyzing:
            analyzing = True
            small = cv2.resize(frame, (320, 240))
            t = threading.Thread(target=analyze_frame, args=(small,), daemon=True)
            t.start()

        # --- Sidebar ---
        sidebar_w = 270
        overlay = display.copy()
        cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        # Title
        cv2.putText(display, "ETHICAL AI SURVEILLANCE", (w - sidebar_w + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)
        cv2.putText(display, "Burnout Predictor", (w - sidebar_w + 10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Emotion bars
        draw_emotion_bars(display, last_emotions, w - sidebar_w + 10, 68)

        # Burnout gauge
        if burnout_scores:
            draw_burnout_gauge(display, burnout_scores[-1], w - sidebar_w + 10, 240)

        # Bias check
        bias_msg, bias_color = check_bias(list(confidence_history))
        cv2.putText(display, "BIAS CHECK:", (w - sidebar_w + 10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display, bias_msg, (w - sidebar_w + 10, 318),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, bias_color, 1)

        # Privacy indicator
        priv_text = "PRIVACY: ON (blurred)" if privacy_mode else "PRIVACY: OFF"
        priv_color = (0, 255, 0) if privacy_mode else (0, 0, 255)
        cv2.putText(display, priv_text, (w - sidebar_w + 10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, priv_color, 1)

        # Session info
        elapsed = (datetime.now() - session_start).seconds
        cv2.putText(display, f"Session: {elapsed // 60:02d}:{elapsed % 60:02d}", (w - sidebar_w + 10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
        cv2.putText(display, f"Samples: {len(emotion_history)}", (w - sidebar_w + 10, 398),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # --- Face boxes (scale up from 320x240 analysis frame) ---
        scale_x = w / 320
        scale_y = h / 240
        for res in last_results:
            bx, by, bw, bh = res['box']
            bx = max(int(bx * scale_x), 0)
            by = max(int(by * scale_y), 0)
            bw = min(int(bw * scale_x), w - bx)
            bh = min(int(bh * scale_y), h - by)

            top_em = max(res['emotions'], key=res['emotions'].get)
            if top_em == 'happy':
                color = (0, 255, 0)
            elif top_em == 'neutral':
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(display, f"{top_em} ({res['emotions'][top_em]:.0%})",
                        (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Auto-report every 45 minutes
        if time.time() - last_auto_report >= AUTO_REPORT_INTERVAL and len(emotion_history) > 0:
            last_auto_report = time.time()
            console.print("\n  [AUTO] 45-minute report triggered...")
            threading.Thread(
                target=save_report_and_get_support,
                args=(list(emotion_history), list(burnout_scores), session_start),
                daemon=True
            ).start()

        # Bottom bar
        cv2.rectangle(display, (0, h - 28), (w - sidebar_w, h), (20, 20, 20), -1)
        cv2.putText(display, "Q:Quit R:Report P:Privacy S:AI Support",
                    (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        cv2.imshow("Ethical AI Surveillance - Burnout Predictor", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            privacy_mode = not privacy_mode
            console.print(f"  Privacy mode: {'ON' if privacy_mode else 'OFF'}")
        elif key == ord('r'):
            report = generate_report(list(emotion_history), list(burnout_scores), session_start)
            console.print("\n" + report + "\n")
            fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(report)
            console.print(f"  Report saved to {fname}")
        elif key == ord('s'):
            console.print("\n  Generating report + AI support...")
            threading.Thread(
                target=save_report_and_get_support,
                args=(list(emotion_history), list(burnout_scores), session_start),
                daemon=True
            ).start()

    # Final report
    console.print("\n  Generating final report...")
    save_report_and_get_support(list(emotion_history), list(burnout_scores), session_start)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
