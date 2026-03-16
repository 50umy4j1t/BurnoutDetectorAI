from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

try:
    import numpy as np
except Exception:
    np = None

try:
    import serial
except Exception:
    serial = None

try:
    import ollama
except Exception:
    ollama = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    from kokoro_onnx import Kokoro
except Exception:
    Kokoro = None


SCRIPT_ROOT = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = APP_ROOT.parent
MAIN_FILE = REPO_ROOT / "main.py"
PYTHON_FROM_VENV = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_PATTERN = "report_*.txt"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
DEFAULT_MODELS = {
    "1": ("qwen3:4b", "Qwen3 4B (Best quality, slower)"),
    "2": ("qwen3:1.7b", "Qwen3 1.7B (Balanced)"),
    "3": ("qwen3:0.6b", "Qwen3 0.6B (Fastest, lightweight)"),
}

HEART_SENSOR_PORT = os.getenv("HEART_SENSOR_PORT", "COM9")
HEART_SENSOR_BAUD = int(os.getenv("HEART_SENSOR_BAUD", "9600"))
HEART_SENSOR_SERIAL_TIMEOUT = float(os.getenv("HEART_SENSOR_SERIAL_TIMEOUT", "1.0"))
HEART_SENSOR_WARMUP_SECONDS = float(os.getenv("HEART_SENSOR_WARMUP_SECONDS", "2.0"))
HEART_SENSOR_MAX_WAIT_SECONDS = float(os.getenv("HEART_SENSOR_MAX_WAIT_SECONDS", "60.0"))

KOKORO_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_MODEL_FILE = "onnx/model.onnx"
KOKORO_VOICE_ID = "af_heart"
KOKORO_VOICE_RAW_FILE = f"voices/{KOKORO_VOICE_ID}.bin"
KOKORO_VOICE_BUNDLE = f"voices-{KOKORO_VOICE_ID}-v1.0.npz"
TTS_TEXT_LIMIT = int(os.getenv("STRESSLENS_TTS_TEXT_LIMIT", "500"))

STDOUT_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
TTS_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

MODEL_CATALOG: dict[str, tuple[str, str]] = {}
SELECTED_MODEL_CHOICE = "1"
MAIN_PROCESS: subprocess.Popen[str] | None = None
KNOWN_REPORTS: dict[str, int] = {}
LAST_HEART_METRICS: dict[str, Any] | None = None
TTS_PIPELINE: Any | None = None
TTS_ENABLED = os.getenv("STRESSLENS_TTS_ENABLED", "1").strip() != "0"
TTS_STATE = ""
TTS_MESSAGE = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_payload(payload: dict[str, Any]) -> None:
    with STDOUT_LOCK:
        sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
        sys.stdout.flush()


def emit_event(event: str, data: dict[str, Any]) -> None:
    write_payload({"type": "event", "event": event, "data": data})


def emit_log(source: str, message: str) -> None:
    cleaned = ANSI_PATTERN.sub("", message or "").strip()
    if not cleaned:
        return
    emit_event(
        "log",
        {
            "source": source,
            "message": cleaned,
            "timestamp": utc_now(),
        },
    )


def respond_success(request_id: int, result: dict[str, Any]) -> None:
    write_payload({"id": request_id, "ok": True, "result": result})


def respond_error(request_id: int, error: str) -> None:
    write_payload({"id": request_id, "ok": False, "error": error})


def python_executable() -> str:
    if PYTHON_FROM_VENV.exists():
        return str(PYTHON_FROM_VENV)
    return sys.executable


def python_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def tts_missing_dependencies() -> list[str]:
    missing: list[str] = []
    if np is None:
        missing.append("numpy")
    if sd is None:
        missing.append("sounddevice")
    if hf_hub_download is None:
        missing.append("huggingface_hub")
    if Kokoro is None:
        missing.append("kokoro_onnx")
    return missing


def tts_available() -> bool:
    return len(tts_missing_dependencies()) == 0


def current_tts_payload() -> dict[str, Any]:
    available = tts_available()
    state = TTS_STATE
    message = TTS_MESSAGE.strip()

    if not available:
        state = "unavailable"
        message = f"Python TTS unavailable: missing {', '.join(tts_missing_dependencies())}"
    elif not TTS_ENABLED and state not in {"loading", "speaking"}:
        state = "muted"
        if not message:
            message = "Python TTS muted"
    elif not state:
        state = "ready"
        if not message:
            message = "Python TTS ready on first use"
    elif not message and state == "ready":
        message = "Python TTS ready"

    return {
        "available": available,
        "enabled": TTS_ENABLED,
        "state": state,
        "message": message,
        "voice": KOKORO_VOICE_ID,
    }


def emit_tts_status(state: str | None = None, message: str | None = None) -> dict[str, Any]:
    global TTS_STATE, TTS_MESSAGE

    if state is not None:
        TTS_STATE = state
    if message is not None:
        TTS_MESSAGE = message

    payload = current_tts_payload()
    emit_event("tts-status", payload)
    return payload


def prepare_kokoro_assets() -> tuple[str, str]:
    if np is None or hf_hub_download is None:
        raise RuntimeError("Kokoro dependencies are not available")

    model_path = hf_hub_download(KOKORO_REPO_ID, KOKORO_MODEL_FILE)
    raw_voice_path = hf_hub_download(KOKORO_REPO_ID, KOKORO_VOICE_RAW_FILE)

    cache_dir = SCRIPT_ROOT / ".cache" / "kokoro"
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
        emit_log("tts", "Preparing Kokoro voice bundle")
        voice_data = np.fromfile(raw_voice_path, dtype=np.float32)
        if voice_data.size % 256 != 0:
            raise ValueError(
                f"Unexpected voice tensor size {voice_data.size} from {KOKORO_VOICE_RAW_FILE}"
            )
        voice_style = voice_data.reshape(-1, 1, 256)
        np.savez(voices_bundle_path, **{KOKORO_VOICE_ID: voice_style})

    return model_path, str(voices_bundle_path)


def apply_kokoro_runtime_compat(pipeline: Any) -> None:
    if np is None:
        return

    input_names = [item.name for item in pipeline.sess.get_inputs()]
    if "input_ids" not in input_names:
        return

    speed_input = next((item for item in pipeline.sess.get_inputs() if item.name == "speed"), None)
    speed_dtype = np.float32 if (speed_input and "float" in speed_input.type) else np.int32

    def _create_audio_compat(self: Any, phonemes: str, voice: Any, speed: float) -> tuple[Any, int]:
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
        return np.asarray(audio, dtype=np.float32).reshape(-1), 24000

    pipeline._create_audio = MethodType(_create_audio_compat, pipeline)


def ensure_tts_pipeline() -> None:
    global TTS_PIPELINE

    if TTS_PIPELINE is not None:
        return

    emit_tts_status("loading", "Loading Kokoro TTS in Python...")
    emit_log("tts", "Loading Kokoro TTS model")
    model_path, voices_path = prepare_kokoro_assets()
    TTS_PIPELINE = Kokoro(model_path, voices_path)
    apply_kokoro_runtime_compat(TTS_PIPELINE)
    emit_tts_status("ready", "Python TTS ready")


def speak_text(text: str, reason: str) -> bool:
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return False

    if not TTS_ENABLED:
        emit_tts_status("muted", "Python TTS muted")
        return False

    if not tts_available():
        payload = emit_tts_status("unavailable", current_tts_payload()["message"])
        emit_log("tts", payload["message"])
        return False

    if np is None or sd is None:
        payload = emit_tts_status("unavailable", current_tts_payload()["message"])
        emit_log("tts", payload["message"])
        return False

    try:
        with TTS_LOCK:
            ensure_tts_pipeline()
            emit_tts_status("speaking", f"Speaking {reason} in Python...")
            samples, sample_rate = TTS_PIPELINE.create(
                cleaned_text[:TTS_TEXT_LIMIT],
                voice=KOKORO_VOICE_ID,
                speed=1.1,
            )
            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            sd.play(samples, samplerate=sample_rate)
            sd.wait()
            emit_tts_status("ready", "Python TTS ready")
            emit_log("tts", f"Finished speaking {reason}")
            return True
    except Exception as exc:
        emit_tts_status("error", f"Python TTS error: {exc}")
        emit_log("tts", f"TTS error: {exc}")
        return False


def queue_tts(text: str, reason: str) -> bool:
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return False

    if not TTS_ENABLED:
        emit_tts_status("muted", "Python TTS muted")
        return False

    if not tts_available():
        payload = emit_tts_status("unavailable", current_tts_payload()["message"])
        emit_log("tts", payload["message"])
        return False

    emit_tts_status("loading", f"Queued Python TTS for {reason}")
    emit_log("tts", f"Queued Python TTS for {reason}")
    threading.Thread(target=speak_text, args=(cleaned_text, reason), daemon=True).start()
    return True


def set_tts_enabled(enabled: Any) -> dict[str, Any]:
    global TTS_ENABLED

    TTS_ENABLED = bool(enabled)
    if not tts_available():
        payload = emit_tts_status("unavailable", current_tts_payload()["message"])
        emit_log("tts", payload["message"])
        return payload

    if TTS_ENABLED:
        payload = emit_tts_status(
            "ready",
            "Python TTS ready on first use" if TTS_PIPELINE is None else "Python TTS ready",
        )
        emit_log("tts", "Python TTS enabled")
        return payload

    payload = emit_tts_status("muted", "Python TTS muted")
    emit_log("tts", "Python TTS muted")
    return payload


def load_model_catalog() -> tuple[dict[str, tuple[str, str]], str]:
    catalog = dict(DEFAULT_MODELS)
    default_choice = "1"

    if not MAIN_FILE.exists():
        return catalog, default_choice

    try:
        module = ast.parse(MAIN_FILE.read_text(encoding="utf-8"))
        available_models = None
        default_model_name = None

        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue

            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "AVAILABLE_MODELS":
                    available_models = ast.literal_eval(node.value)
                if target.id == "OLLAMA_MODEL":
                    default_model_name = ast.literal_eval(node.value)

        if isinstance(available_models, dict):
            catalog = {
                str(choice): (str(values[0]), str(values[1]))
                for choice, values in available_models.items()
            }

        if default_model_name:
            for choice, values in catalog.items():
                if values[0] == default_model_name:
                    default_choice = choice
                    break
    except Exception as exc:
        emit_log("bridge", f"Model catalog fallback: {exc}")

    return catalog, default_choice


def model_payload() -> list[dict[str, str]]:
    payload = []
    for choice, (model_name, description) in MODEL_CATALOG.items():
        payload.append(
            {
                "choice": choice,
                "model": model_name,
                "description": description,
                "label": f"{choice}. {description}",
            }
        )
    payload.sort(key=lambda item: item["choice"])
    return payload


def normalize_model_choice(raw_choice: Any) -> str:
    choice = str(raw_choice or SELECTED_MODEL_CHOICE).strip()
    if choice in MODEL_CATALOG:
        return choice
    return SELECTED_MODEL_CHOICE if SELECTED_MODEL_CHOICE in MODEL_CATALOG else "1"


def report_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "fileName": path.name,
        "modifiedAt": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sizeBytes": stat.st_size,
    }


def collect_reports() -> tuple[dict[str, int], list[dict[str, Any]]]:
    snapshot: dict[str, int] = {}
    report_paths = sorted(
        REPO_ROOT.glob(REPORT_PATTERN),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    reports: list[dict[str, Any]] = []
    for path in report_paths:
        stat = path.stat()
        snapshot[path.name] = stat.st_mtime_ns
        reports.append(report_metadata(path))

    return snapshot, reports


def latest_report_payload() -> dict[str, Any] | None:
    _, reports = collect_reports()
    if not reports:
        return None

    latest = reports[0]
    latest["content"] = read_report_text(latest["fileName"])
    return latest


def resolve_report_path(file_name: str) -> Path:
    candidate = (REPO_ROOT / file_name).resolve()
    repo_root = REPO_ROOT.resolve()

    if candidate.parent != repo_root:
        raise ValueError("Report path must stay inside the repository root")
    if not candidate.name.startswith("report_") or candidate.suffix.lower() != ".txt":
        raise ValueError("Invalid report file")
    if not candidate.exists():
        raise FileNotFoundError(f"Report not found: {file_name}")

    return candidate


def read_report_text(file_name: str) -> str:
    return resolve_report_path(file_name).read_text(encoding="utf-8")


def parse_heart_sensor_line(line: str) -> dict[str, int] | None:
    values: dict[str, str] = {}
    for part in line.split(","):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        values[key.strip().lower()] = raw_value.strip().rstrip("%")

    bpm_text = values.get("bpm")
    spo2_text = values.get("spo2")
    if not bpm_text or not spo2_text:
        return None

    try:
        bpm = int(float(bpm_text))
        spo2 = int(float(spo2_text))
    except ValueError:
        return None

    return {"bpm": bpm, "spo2": spo2}


def capture_heart_metrics() -> dict[str, Any]:
    global LAST_HEART_METRICS

    if serial is None:
        raise RuntimeError("pyserial is not available in the Python environment")

    emit_log("heart", f"Listening on {HEART_SENSOR_PORT} @ {HEART_SENSOR_BAUD} baud")
    emit_log("heart", "Place your finger on the sensor and hold still")

    try:
        with serial.Serial(
            HEART_SENSOR_PORT,
            HEART_SENSOR_BAUD,
            timeout=HEART_SENSOR_SERIAL_TIMEOUT,
        ) as connection:
            time.sleep(HEART_SENSOR_WARMUP_SECONDS)
            started_at = time.monotonic()

            while True:
                if HEART_SENSOR_MAX_WAIT_SECONDS > 0:
                    if (time.monotonic() - started_at) >= HEART_SENSOR_MAX_WAIT_SECONDS:
                        raise RuntimeError("Heart sensor timed out before a reading was received")

                raw_line = connection.readline().decode(errors="ignore").strip()
                if not raw_line:
                    continue

                metrics = parse_heart_sensor_line(raw_line)
                if not metrics:
                    continue

                metrics["capturedAt"] = utc_now()
                metrics["port"] = HEART_SENSOR_PORT
                LAST_HEART_METRICS = metrics
                emit_event("heart-rate", metrics)
                return metrics
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def stream_main_output(stream: Any, source: str) -> None:
    try:
        while True:
            line = stream.readline()
            if not line:
                break
            emit_log(source, line)
    except Exception as exc:
        emit_log(source, f"Output reader stopped: {exc}")


def wait_for_main_exit(process: subprocess.Popen[str]) -> None:
    global MAIN_PROCESS

    exit_code = process.wait()
    with STATE_LOCK:
        if MAIN_PROCESS is process:
            MAIN_PROCESS = None

    emit_event(
        "main-status",
        {
            "running": False,
            "exitCode": exit_code,
            "modelChoice": SELECTED_MODEL_CHOICE,
            "modelName": MODEL_CATALOG.get(SELECTED_MODEL_CHOICE, ("", ""))[0],
        },
    )
    emit_log("main", f"main.py exited with code {exit_code}")


def launch_main_process(model_choice: str) -> dict[str, Any]:
    global MAIN_PROCESS, SELECTED_MODEL_CHOICE

    with STATE_LOCK:
        if MAIN_PROCESS and MAIN_PROCESS.poll() is None:
            return {
                "running": True,
                "alreadyRunning": True,
                "pid": MAIN_PROCESS.pid,
                "modelChoice": SELECTED_MODEL_CHOICE,
                "modelName": MODEL_CATALOG.get(SELECTED_MODEL_CHOICE, ("", ""))[0],
            }

        SELECTED_MODEL_CHOICE = normalize_model_choice(model_choice)

        process = subprocess.Popen(
            [python_executable(), "-u", str(MAIN_FILE)],
            cwd=str(REPO_ROOT),
            env=python_subprocess_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        MAIN_PROCESS = process

    threading.Thread(
        target=stream_main_output,
        args=(process.stdout, "main:stdout"),
        daemon=True,
    ).start()
    threading.Thread(
        target=stream_main_output,
        args=(process.stderr, "main:stderr"),
        daemon=True,
    ).start()
    threading.Thread(target=wait_for_main_exit, args=(process,), daemon=True).start()

    time.sleep(0.25)
    if process.stdin:
        process.stdin.write(f"{SELECTED_MODEL_CHOICE}\n")
        process.stdin.flush()

    model_name = MODEL_CATALOG.get(SELECTED_MODEL_CHOICE, ("", ""))[0]
    emit_event(
        "main-status",
        {
            "running": True,
            "pid": process.pid,
            "modelChoice": SELECTED_MODEL_CHOICE,
            "modelName": model_name,
        },
    )
    emit_log("main", f"Launched main.py with model choice {SELECTED_MODEL_CHOICE} ({model_name})")

    return {
        "running": True,
        "pid": process.pid,
        "modelChoice": SELECTED_MODEL_CHOICE,
        "modelName": model_name,
    }


def terminate_main_process() -> dict[str, Any]:
    global MAIN_PROCESS

    with STATE_LOCK:
        process = MAIN_PROCESS

    if not process or process.poll() is not None:
        return {"running": False, "note": "main.py is not running"}

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    with STATE_LOCK:
        if MAIN_PROCESS is process:
            MAIN_PROCESS = None

    return {
        "running": False,
        "note": "main.py was force-stopped. Use Q inside the camera window for a clean final report.",
    }


def bootstrap_payload() -> dict[str, Any]:
    _, reports = collect_reports()
    latest_report = latest_report_payload()
    with STATE_LOCK:
        main_running = bool(MAIN_PROCESS and MAIN_PROCESS.poll() is None)
        main_pid = MAIN_PROCESS.pid if main_running and MAIN_PROCESS else None

    return {
        "models": model_payload(),
        "selectedModelChoice": SELECTED_MODEL_CHOICE,
        "reports": reports,
        "latestReport": latest_report,
        "latestHeartMetrics": LAST_HEART_METRICS,
        "heartSensor": {
            "available": serial is not None,
            "port": HEART_SENSOR_PORT,
            "baud": HEART_SENSOR_BAUD,
            "timeoutSeconds": HEART_SENSOR_MAX_WAIT_SECONDS,
        },
        "mainRunning": {
            "running": main_running,
            "pid": main_pid,
        },
        "tts": current_tts_payload(),
        "pythonExecutable": python_executable(),
        "reportDirectory": str(REPO_ROOT),
        "launchNote": "Use Q, R, P, and S inside the camera window. Force Stop skips the clean final report flow.",
    }


def list_reports_payload() -> dict[str, Any]:
    _, reports = collect_reports()
    return {"reports": reports}


def read_report_payload(file_name: str) -> dict[str, Any]:
    content = read_report_text(file_name)
    metadata = report_metadata(resolve_report_path(file_name))
    metadata["content"] = content
    return metadata


def chat_with_ollama(message: str, include_latest_report: bool, model_choice: str) -> dict[str, Any]:
    global SELECTED_MODEL_CHOICE

    if ollama is None:
        raise RuntimeError("The Python ollama package is not available in the environment")

    cleaned_message = str(message or "").strip()
    if not cleaned_message:
        raise ValueError("A chat message is required")

    SELECTED_MODEL_CHOICE = normalize_model_choice(model_choice)
    model_name = MODEL_CATALOG.get(SELECTED_MODEL_CHOICE, ("", ""))[0]

    latest_report = latest_report_payload() if include_latest_report else None
    user_parts = [cleaned_message]
    if latest_report:
        user_parts.append(
            f"Latest session report ({latest_report['fileName']}):\n\n{latest_report['content']}"
        )

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a calm and practical wellness advisor inside a local desktop app. "
                    "Keep replies concise, grounded, and supportive. "
                    "Use 3 to 6 sentences. If a session report is provided, base your advice on it."
                ),
            },
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        think=True,
    )

    message_block = getattr(response, "message", response)
    reply_text = getattr(message_block, "content", None)
    if reply_text is None and isinstance(message_block, dict):
        reply_text = message_block.get("content")

    reasoning_text = getattr(message_block, "thinking", None)
    if reasoning_text is None and isinstance(message_block, dict):
        reasoning_text = message_block.get("thinking")

    return {
        "response": reply_text or "",
        "thinking": reasoning_text,
        "model": model_name,
        "includedReport": latest_report["fileName"] if latest_report else None,
        "tts": current_tts_payload(),
    }


def speak_text_payload(text: str, reason: str | None) -> dict[str, Any]:
    queued = queue_tts(text, reason or "chat reply")
    return {
        "queued": queued,
        "tts": current_tts_payload(),
    }


def watch_reports() -> None:
    global KNOWN_REPORTS

    KNOWN_REPORTS, _ = collect_reports()

    while not STOP_EVENT.wait(1.5):
        snapshot, reports = collect_reports()
        if snapshot == KNOWN_REPORTS:
            continue

        previous = KNOWN_REPORTS
        KNOWN_REPORTS = snapshot

        latest_file_name = reports[0]["fileName"] if reports else None
        change_kind = "updated"
        if latest_file_name and latest_file_name not in previous:
            change_kind = "created"

        payload: dict[str, Any] = {
            "kind": change_kind,
            "fileName": latest_file_name,
            "reports": reports,
        }
        if latest_file_name:
            payload["content"] = read_report_text(latest_file_name)

        emit_event("report-updated", payload)


def handle_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "bootstrap":
        return bootstrap_payload()
    if method == "launch_main":
        return launch_main_process(params.get("modelChoice"))
    if method == "terminate_main":
        return terminate_main_process()
    if method == "capture_heart_rate":
        return {"metrics": capture_heart_metrics()}
    if method == "set_tts_enabled":
        return set_tts_enabled(params.get("enabled"))
    if method == "speak_text":
        return speak_text_payload(params.get("text"), params.get("reason"))
    if method == "chat":
        return chat_with_ollama(
            params.get("message"),
            bool(params.get("includeLatestReport")),
            params.get("modelChoice"),
        )
    if method == "list_reports":
        return list_reports_payload()
    if method == "read_report":
        file_name = str(params.get("fileName") or "").strip()
        if not file_name:
            raise ValueError("fileName is required")
        return read_report_payload(file_name)
    if method == "shutdown":
        STOP_EVENT.set()
        return {"ok": True}

    raise ValueError(f"Unsupported bridge method: {method}")


def main() -> None:
    global MODEL_CATALOG, SELECTED_MODEL_CHOICE

    MODEL_CATALOG, SELECTED_MODEL_CHOICE = load_model_catalog()
    threading.Thread(target=watch_reports, daemon=True).start()
    emit_log("bridge", f"Bridge ready using {python_executable()}")

    for raw_line in sys.stdin:
        if STOP_EVENT.is_set():
            break

        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit_log("bridge", f"Invalid JSON request: {exc}")
            continue

        request_id = int(request.get("id", 0))
        method = str(request.get("method") or "").strip()
        params = request.get("params") or {}

        try:
            result = handle_request(method, params)
            if request_id > 0:
                respond_success(request_id, result)
        except Exception as exc:
            emit_log("bridge", f"{method or 'unknown'} failed: {exc}")
            if request_id > 0:
                respond_error(request_id, str(exc))

        if method == "shutdown":
            break


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass