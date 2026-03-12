# Frontend Developer Guide

This document describes the FastAPI backend API and how to integrate it into a React frontend.

## Running the Backend

```bash
pip install fastapi uvicorn[standard] opencv-python numpy deepface tensorflow ollama rich
python server.py
# Backend runs at http://localhost:8000
```

Requires [Ollama](https://ollama.com/) running locally with at least one Qwen3 model pulled:
```bash
ollama pull qwen3:4b
```

---

## API Reference

### Models

#### `GET /api/models`
Returns available LLM models and the currently selected one.

**Response:**
```json
{
  "models": [
    { "id": "qwen3:4b", "name": "Qwen3 4B (Best quality, slower)" },
    { "id": "qwen3:1.7b", "name": "Qwen3 1.7B (Balanced)" },
    { "id": "qwen3:0.6b", "name": "Qwen3 0.6B (Fastest, lightweight)" }
  ],
  "current": "qwen3:4b"
}
```

---

### Session Management

#### `POST /api/session/start`
Opens the webcam and begins emotion detection.

**Request body:**
```json
{ "model": "qwen3:4b" }  // optional, defaults to qwen3:4b
```

**Response (200):**
```json
{ "status": "started", "model": "qwen3:4b" }
```

**Error (400):** `{ "error": "Session already running" }`
**Error (500):** `{ "error": "Cannot open webcam" }`

#### `POST /api/session/stop`
Stops the webcam session and returns a final report.

**Response (200):**
```json
{ "status": "stopped", "report": "======= SESSION REPORT ======\n..." }
```

#### `POST /api/settings`
Update session settings (can be called while session is running).

**Request body:**
```json
{
  "privacy_mode": true,   // optional - toggle webcam blur
  "model": "qwen3:1.7b"   // optional - change LLM model
}
```

**Response:** `{ "ok": true }`

---

### Video Feed

#### `GET /api/video_feed`
MJPEG streaming endpoint. Returns a continuous stream of JPEG frames with overlays (emotion bars, face boxes, burnout gauge drawn on the video).

**Usage in React:**
```jsx
// Simply use an img tag — the browser handles MJPEG natively
<img src="/api/video_feed" alt="Live Feed" />
```

The feed includes server-rendered overlays. No client-side processing needed for the video.

---

### WebSocket — Live Emotion Data

#### `WS /ws/emotions`
Pushes emotion data as JSON every ~300ms while a session is running.

**Message format:**
```json
{
  "emotions": {
    "happy": 0.05,
    "sad": 0.12,
    "angry": 0.03,
    "fear": 0.02,
    "surprise": 0.01,
    "disgust": 0.01,
    "neutral": 0.76
  },
  "burnout": 23.5,
  "bias": "Stable (0.72 +/-0.08)",
  "samples": 142,
  "elapsed": 347,
  "privacy": true,
  "running": true,
  "model": "qwen3:4b"
}
```

**Field details:**
| Field | Type | Description |
|-------|------|-------------|
| `emotions` | `Record<string, number>` | 7 emotion scores, each 0.0-1.0, sum to ~1.0 |
| `burnout` | `number` | Burnout risk percentage (0-100) |
| `bias` | `string` | Bias check status message |
| `samples` | `number` | Total emotion samples collected |
| `elapsed` | `number` | Session duration in seconds |
| `privacy` | `boolean` | Whether privacy blur is active |
| `running` | `boolean` | Whether session is active |
| `model` | `string` | Current LLM model ID |

**Usage in React:**
```jsx
useEffect(() => {
  const ws = new WebSocket(`ws://${window.location.host}/ws/emotions`);
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    setEmotions(data.emotions);
    setBurnout(data.burnout);
    // ...
  };
  return () => ws.close();
}, []);
```

---

### Reports & Chat

#### `POST /api/report`
Generate a text report from current session data (does NOT call AI).

**Response:**
```json
{ "report": "======= SESSION REPORT ======\n..." }
```

#### `POST /api/support`
Generate a report AND send it to the AI wellness advisor in one call.

**Response:**
```json
{
  "report": "======= SESSION REPORT ======\n...",
  "thinking": "The student shows elevated sadness...",   // optional, may be absent
  "response": "I can see you've been working hard today..."
}
```

#### `POST /api/chat`
Send a free-text message to the AI advisor. Session context is automatically included.

**Request body:**
```json
{ "message": "I'm feeling tired, should I take a break?" }
```

**Response:**
```json
{
  "thinking": "Based on the session data...",   // optional
  "response": "It sounds like you could use a short break..."
}
```

The backend maintains chat history (last 20 messages) for multi-turn conversation.

---

## Text-to-Speech (Browser-Side)

TTS runs entirely in the browser using [Kokoro.js](https://www.npmjs.com/package/kokoro-js). The backend does NOT handle TTS.

### Installation
```bash
npm i kokoro-js
```

### React Integration
```jsx
import { KokoroTTS } from 'kokoro-js';
import { useRef, useCallback } from 'react';

function useTTS() {
  const ttsRef = useRef(null);
  const loadingRef = useRef(false);

  const init = useCallback(async () => {
    if (ttsRef.current || loadingRef.current) return ttsRef.current;
    loadingRef.current = true;
    ttsRef.current = await KokoroTTS.from_pretrained(
      'onnx-community/Kokoro-82M-ONNX',
      { dtype: 'q8' }  // ~86MB, cached by browser after first download
    );
    loadingRef.current = false;
    return ttsRef.current;
  }, []);

  const speak = useCallback(async (text) => {
    const engine = await init();
    if (!engine) return;
    const audio = await engine.generate(text, { voice: 'af_heart' });
    await audio.play();
  }, [init]);

  return { speak, init };
}
```

### Available Voices
Call `tts.list_voices()` after initialization to get the full list. Some options:
- `af_heart` — warm female voice (recommended)
- `af_sky` — female voice
- `am_adam` — male voice

### Quantization Options
| dtype | Size | Quality |
|-------|------|---------|
| `fp32` | 326 MB | Best |
| `q8` | 86 MB | No noticeable loss (recommended) |
| `q4` | ~50 MB | Slight quality loss |

---

## Layout Specification

The UI has 4 main panels in a 2x2 grid:

```
+----------------------------------+--------------------+
|                                  | Emotions Panel     |
|   Video Feed                     |  - 7 emotion bars  |
|   (img → /api/video_feed)        |  - burnout gauge   |
|                                  |  - bias check      |
|                                  |  - session stats   |
+----------------------------------+--------------------+
|  Heart Rate Panel                | AI Chat Panel      |
|  (placeholder for now)           |  - message list    |
|                                  |  - report display  |
|                                  |  - text input      |
|                                  |  - TTS controls    |
+----------------------------------+--------------------+
```

### Top Bar
- App title
- Model selector dropdown (from `GET /api/models`)
- Start / Stop session buttons
- Privacy mode toggle
- Mute TTS toggle

### Video Feed Panel (top-left)
- `<img src="/api/video_feed">` — no JS needed
- Show placeholder when session is not running

### Emotions Panel (top-right)
- Data from WebSocket `/ws/emotions`
- Horizontal bars for each of 7 emotions (sorted by value descending)
- Burnout gauge (0-100%, color-coded: green/yellow/orange/red)
- Bias check status text
- Sample count and session elapsed time

### Heart Rate Panel (bottom-left)
- **Placeholder** — will be integrated with external sensor data in the future
- Show "-- BPM" and a flatline animation
- Expected future API shape (not yet implemented):
  ```
  WS /ws/heartrate → { "bpm": 72, "timestamp": "...", "status": "connected" }
  ```

### AI Chat Panel (bottom-right)
- "Report + AI Support" button → calls `POST /api/support`
- Shows report in a monospace block
- Shows AI responses as chat bubbles
- Optionally shows AI "thinking" in a collapsed/dimmed section
- Free-text input → calls `POST /api/chat`
- Auto-speaks AI responses via Kokoro.js (unless muted)

---

## CORS / Auth

- Currently no authentication or CORS restrictions (localhost only)
- When deploying, add CORS middleware in `server.py`:
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], ...)
  ```

---

## Notes

- Only one webcam session can run at a time (single webcam consumer)
- The MJPEG feed is ~1-1.5 MB/s at 30fps — fine for localhost
- DeepFace loads its model on first analysis call (may take a few seconds)
- The Kokoro TTS model (~86MB) downloads on first browser use and is cached
- All processing (emotion detection, LLM inference, TTS) runs locally — nothing is sent to external servers
