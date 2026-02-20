# Ethical AI Surveillance System - Smart Academic Burnout Predictor

**Team: Project Protocol**

A real-time webcam-based system that monitors facial emotions to predict academic burnout in students. All processing is done locally with privacy-first design — no facial data is stored, and frames can be blurred in real-time.

## Features

- **Real-time Emotion Detection** — Uses DeepFace with OpenCV to analyze facial expressions (happy, sad, angry, fear, surprise, disgust, neutral)
- **Burnout Risk Scoring** — Calculates a burnout score based on negative emotion patterns over time
- **AI Wellness Advisor** — Sends session reports to a local Ollama model (Qwen3) for supportive, personalized feedback
- **Text-to-Speech** — Reads AI responses aloud using Sarvam AI TTS
- **Privacy Mode** — Gaussian blur applied to the live feed by default; no raw frames are ever saved
- **Bias Detection** — Monitors confidence variance to flag potential model bias or poor lighting
- **Auto Reports** — Automatically generates a wellness report every 45 minutes
- **Model Selection** — Choose between Qwen3 4B, 1.7B, or 0.6B at startup based on your hardware

## Requirements

- Python 3.10+
- Webcam
- [Ollama](https://ollama.com/) running locally with a Qwen3 model pulled

## Setup

1. Install dependencies:
   ```bash
   pip install opencv-python numpy deepface rich ollama requests tensorflow
   ```

2. Pull an Ollama model:
   ```bash
   ollama pull qwen3:4b
   # or for lighter models:
   ollama pull qwen3:1.7b
   ollama pull qwen3:0.6b
   ```

3. (Optional) Set your Sarvam API key for TTS:
   ```bash
   export SARVAM_KEY="your-api-key"
   ```

4. Run:
   ```bash
   python main.py
   ```

## Controls

| Key | Action                        |
|-----|-------------------------------|
| `Q` | Quit and generate final report |
| `R` | Generate and save a report     |
| `P` | Toggle privacy mode (blur)     |
| `S` | Get AI wellness support        |

## Privacy

- All emotion analysis runs locally on your machine
- The webcam feed is blurred by default (privacy mode ON)
- No facial images are stored or transmitted
- Only aggregated emotion statistics are saved in text reports
