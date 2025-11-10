# 🎙️ RTVC-Lite: Real-Time Voice Cloning (SV2TTS Architecture)

RTVC-Lite is a simplified and modular implementation of the **SV2TTS** pipeline used for real-time voice cloning.  
Given a short reference audio sample (3–5 seconds), the system can generate speech in the same voice for any text input.

This project is designed for **learning, experiments, academic demonstration, and interview portfolio use** — using clean, readable Python modules that can be easily replaced with real pretrained models.

---

## 🔥 Pipeline Overview

| Stage | Module | Purpose |
|------|--------|---------|
| **Speaker Encoder** | GE2E / encoder module | Extracts a unique **speaker embedding** from reference audio |
| **Synthesizer** | Tacotron-like synthesizer | Generates **mel-spectrograms** from text + voice embedding |
| **Vocoder** | WaveRNN / HiFi-GAN / fallback Griffin-Lim | Converts mel-spectrograms → final speech waveform |

This follows the original SV2TTS structure:

---

## 🛠️ Tech Stack

| Component | Tools |
|---------|--------|
| Programming | Python |
| Deep Learning | PyTorch |
| Audio Processing | Librosa, NumPy, SciPy |
| Models (Replaceable) | GE2E Encoder, Tacotron Synthesizer, WaveRNN / HiFi-GAN Vocoder |
| CLI & Utilities | Click, SoundFile, Matplotlib (optional) |

---

---

## 🚀 Quick Start

### 1) Install dependencies (recommended: uv)
```bash
uv pip install -e .


python demo_cli.py \
  --reference path/to/reference_voice.wav \
  --text "This is a cloned voice demonstration." \
  --out cloned.wav


--encoder-ckpt encoder.pt
--synth-ckpt synthesizer.pt


✅ Replaceable Model Architecture

This repo is structured so you can plug in better models at any time:

Swap Tacotron with FastSpeech / VITS

Swap WaveRNN with HiFi-GAN / WaveGlow

Train or plug in your own speaker encoder

No codebase restructuring required.

🎨 Use Cases

Voice Assistants

Game / Character Voice Design

Personalized TTS Systems

Research and Academic Demonstrations

Accessibility & Speech Aid Tools

⚠️ Responsible Use

Voice cloning can be misused.
You must have permission from the speaker whose voice is being cloned.
Use ethically and disclose when generated voices are synthetic.

⭐ Future Enhancements

Upgrade to FastSpeech2 for faster synthesis

Replace Griffin-Lim fallback with HiFi-GAN

Web UI via Streamlit / Gradio

🤝 Credits

Inspired by the original research:
Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (ArXiv: 1806.04558)

This project is a lightweight educational implementation, not a full reproduction of the original repository.