# 🎧 Multilingual Speech Synthesis with Voice Cloning

An end-to-end AI-powered audio dubbing system that converts English speech into Arabic while preserving the original speaker’s tone, timing, and characteristics using ASR, NMT, TTS, and voice conversion.

---

## 🏗️ Architecture
![Architecture](assets/flow.png)

System Design: FastAPI-based orchestration with GPU-accelerated ASR, TTS, and voice conversion modules, supported by segment-level processing and full observability.

---

## 🚀 Overview
Pipeline:
Audio → ASR → Translation → TTS → Voice Conversion → Alignment → Final Output

---

## ⚙️ Core Components
- ASR: Whisper
- NMT: Argos Translate
- TTS: XTTS (Coqui)
- Voice Conversion: Seed-VC
- Backend: FastAPI
- Framework: PyTorch

---

## ✨ Features
- Cross-lingual voice cloning (English → Arabic)
- Speaker tone preservation using Seed-VC
- Segment-level audio alignment
- Near real-time inference (~0.8–1.2 RTF)
- Metrics tracking (latency, sync error, GPU usage)
- API-ready architecture (FastAPI)
- Structured logging with JSON outputs

---

## ⚙️ Installation
1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt

Requirements:
- Python 3.11
- GPU (recommended)
- FFmpeg

---

## ▶️ Usage
python src/main.py

Output:
- output/arabic_output.mp3
- output/metrics_<run_id>.json

---

## 🌐 API
Run:
uvicorn src.api:app --reload

POST /dub

---

## 📊 Metrics
- RTF
- Latency
- Sync Error
- Success Rate
- GPU Usage

---

## 🧠 Highlights
- Transformer-based ASR + NMT
- XTTS + Seed-VC
- Segment alignment
- GPU acceleration

---

## 💼 Use Cases
- Video dubbing
- Podcast localization
- Multilingual content
- Voice cloning

---

## 🚀 Future Work
- Lip-sync
- Streaming
- Multi-speaker
- Diffusion models

---

## 👤 Author
Shivam Bhatt

---

## 📄 License
MIT

---

## 🙏 Acknowledgement
Inspired in part by:

Transform Voices with AI: A Complete Guide to Seed-VC (LevelUp GitConnected)

This resource provided insights into modern voice conversion techniques, particularly Seed-VC, which influenced the design of the tone-matching and speaker-preserving components of this system.