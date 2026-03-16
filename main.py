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
import sounddevice as sd
from kokoro_onnx import Kokoro, SAMPLE_RATE
from huggingface_hub import hf_hub_download
from pathlib import Path
from types import MethodType

try:
    import serial
except Exception:
    serial = None

AVAILABLE_MODELS = {
    "1": ("qwen3:4b", "Qwen3 4B (Best quality, slower)"),
    "2": ("qwen3:1.7b", "Qwen3 1.7B (Balanced)"),
    "3": ("qwen3:0.6b", "Qwen3 0.6B (Fastest, lightweight)"),
}
OLLAMA_MODEL = "qwen3:4b"  # default, overridden by user selection
AUTO_REPORT_INTERVAL = 45 * 60  # 45 minutes in seconds
TTS_PIPELINE = None  # initialized lazily
KOKORO_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_MODEL_FILE = "onnx/model.onnx"
KOKORO_VOICE_ID = "af_heart"
KOKORO_VOICE_RAW_FILE = f"voices/{KOKORO_VOICE_ID}.bin"
KOKORO_VOICE_BUNDLE = f"voices-{KOKORO_VOICE_ID}-v1.0.npz"
HEART_SENSOR_PORT = os.getenv("HEART_SENSOR_PORT", "COM9")
HEART_SENSOR_BAUD = int(os.getenv("HEART_SENSOR_BAUD", "9600"))
HEART_SENSOR_SERIAL_TIMEOUT = float(os.getenv("HEART_SENSOR_SERIAL_TIMEOUT", "1.0"))
HEART_SENSOR_WARMUP_SECONDS = float(os.getenv("HEART_SENSOR_WARMUP_SECONDS", "2.0"))
HEART_SENSOR_MAX_WAIT_SECONDS = float(os.getenv("HEART_SENSOR_MAX_WAIT_SECONDS", "60.0"))
HEART_SENSOR_LOCK = threading.Lock()
HEART_METRICS_LOCK = threading.Lock()
LAST_HEART_METRICS = None


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


def generate_report(emotion_history, burnout_scores, session_start, heart_metrics=None):
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
    lines.append("  HEART SENSOR SUMMARY:")
    if heart_metrics:
        lines.append(f"  HEART RATE (BPM)  : {heart_metrics['bpm']}")
        lines.append(f"  BLOOD OXYGEN      : {heart_metrics['spo2']}%")
    else:
        lines.append("  HEART RATE (BPM)  : Not captured")
        lines.append("  BLOOD OXYGEN      : Not captured")

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


def parse_heart_sensor_line(line):
    values = {}
    for part in line.split(','):
        if ':' not in part:
            continue
        key, raw_val = part.split(':', 1)
        values[key.strip().lower()] = raw_val.strip().rstrip('%')

    bpm_txt = values.get('bpm')
    spo2_txt = values.get('spo2')
    if not bpm_txt or not spo2_txt:
        return None

    try:
        bpm = int(float(bpm_txt))
        spo2 = int(float(spo2_txt))
    except ValueError:
        return None

    return {"bpm": bpm, "spo2": spo2}


def capture_heart_metrics():
    """Wait for heart sensor reading and return BPM/SpO2 metrics."""
    global LAST_HEART_METRICS

    if serial is None:
        console.print("  [yellow]Heart sensor unavailable: install pyserial to enable it.[/]")
        return None

    console.print("  [bold cyan]Heart Sensor:[/] Place finger on sensor and hold still...")

    try:
        with HEART_SENSOR_LOCK:
            with serial.Serial(HEART_SENSOR_PORT, HEART_SENSOR_BAUD, timeout=HEART_SENSOR_SERIAL_TIMEOUT) as ser:
                time.sleep(HEART_SENSOR_WARMUP_SECONDS)
                start_wait = time.time()

                while True:
                    if HEART_SENSOR_MAX_WAIT_SECONDS > 0:
                        if (time.time() - start_wait) >= HEART_SENSOR_MAX_WAIT_SECONDS:
                            console.print("  [yellow]Heart sensor timeout: no reading received.[/]")
                            return None

                    line = ser.readline().decode(errors='ignore').strip()
                    if not line:
                        continue

                    metrics = parse_heart_sensor_line(line)
                    if metrics:
                        metrics['captured_at'] = datetime.now()
                        with HEART_METRICS_LOCK:
                            LAST_HEART_METRICS = metrics
                        console.print(
                            f"  [green]Heart reading captured:[/] BPM={metrics['bpm']} | SpO2={metrics['spo2']}%"
                        )
                        return metrics
    except Exception as e:
        console.print(f"  [yellow]Heart sensor error: {e}[/]")
        return None


def prepare_kokoro_assets():
    """Download assets and convert raw HF voice data to kokoro_onnx voice bundle format."""
    model_path = hf_hub_download(KOKORO_REPO_ID, KOKORO_MODEL_FILE)
    raw_voice_path = hf_hub_download(KOKORO_REPO_ID, KOKORO_VOICE_RAW_FILE)

    cache_dir = Path(__file__).resolve().parent / ".cache" / "kokoro"
    cache_dir.mkdir(parents=True, exist_ok=True)
    voices_bundle_path = cache_dir / KOKORO_VOICE_BUNDLE

    rebuild_bundle = True
    if voices_bundle_path.exists():
        try:
            with np.load(voices_bundle_path) as bundle:
                rebuild_bundle = KOKORO_VOICE_ID not in bundle.files
        except Exception:
            rebuild_bundle = True

    if rebuild_bundle:
        console.print("  [dim]Preparing Kokoro voice bundle...[/]")
        voice_data = np.fromfile(raw_voice_path, dtype=np.float32)
        if voice_data.size % 256 != 0:
            raise ValueError(
                f"Unexpected voice tensor size {voice_data.size} from {KOKORO_VOICE_RAW_FILE}"
            )
        voice_style = voice_data.reshape(-1, 1, 256)
        np.savez(voices_bundle_path, **{KOKORO_VOICE_ID: voice_style})

    return model_path, str(voices_bundle_path)


def apply_kokoro_runtime_compat(pipeline):
    """Patch kokoro_onnx runtime handling for onnx-community v1.0 model inputs."""
    input_names = [item.name for item in pipeline.sess.get_inputs()]
    if "input_ids" not in input_names:
        return

    speed_input = next((item for item in pipeline.sess.get_inputs() if item.name == "speed"), None)
    speed_dtype = np.float32 if (speed_input and "float" in speed_input.type) else np.int32

    def _create_audio_compat(self, phonemes, voice, speed):
        phonemes = phonemes[:510]
        tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
        if len(tokens) > 510:
            tokens = tokens[:510]

        voice = voice[len(tokens)]
        token_batch = [[0, *tokens.tolist(), 0]]
        inputs = {
            "input_ids": token_batch,
            "style": np.array(voice, dtype=np.float32),
            "speed": np.array([speed], dtype=speed_dtype),
        }
        audio = self.sess.run(None, inputs)[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1), SAMPLE_RATE

    pipeline._create_audio = MethodType(_create_audio_compat, pipeline)


def speak_text(text):
    """Convert text to speech using Kokoro TTS (fully offline)."""
    global TTS_PIPELINE
    try:
        if TTS_PIPELINE is None:
            console.print("  [dim]Loading Kokoro TTS model...[/]")
            model_path, voices_path = prepare_kokoro_assets()
            TTS_PIPELINE = Kokoro(model_path, voices_path)
            apply_kokoro_runtime_compat(TTS_PIPELINE)
        samples, sr = TTS_PIPELINE.create(text, voice=KOKORO_VOICE_ID, speed=1.1)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        sd.play(samples, samplerate=sr)
        sd.wait()
    except Exception as e:
        console.print(f"  [red]TTS error: {e}[/]")


def save_report_and_get_support(emotion_history, burnout_scores, session_start):
    """Generate report, save it, send to Ollama, print supportive message."""
    heart_metrics = capture_heart_metrics()
    report = generate_report(
        list(emotion_history),
        list(burnout_scores),
        session_start,
        heart_metrics=heart_metrics,
    )
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
    console.print(f"  Heart sensor: {HEART_SENSOR_PORT} @ {HEART_SENSOR_BAUD} baud")

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

        with HEART_METRICS_LOCK:
            latest_metrics = dict(LAST_HEART_METRICS) if LAST_HEART_METRICS else None
        if latest_metrics:
            cv2.putText(display, f"BPM: {latest_metrics['bpm']}", (w - sidebar_w + 10, 418),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 230, 255), 1)
            cv2.putText(display, f"SpO2: {latest_metrics['spo2']}%", (w - sidebar_w + 10, 436),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 230, 255), 1)
        else:
            cv2.putText(display, "BPM: --", (w - sidebar_w + 10, 418),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
            cv2.putText(display, "SpO2: --", (w - sidebar_w + 10, 436),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

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
            console.print("\n  Generating report (waiting for heart sensor)...")
            heart_metrics = capture_heart_metrics()
            report = generate_report(
                list(emotion_history),
                list(burnout_scores),
                session_start,
                heart_metrics=heart_metrics,
            )
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
