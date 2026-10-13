# 🎙️ Automatic Speech Recognition (ASR) Tool

An end-to-end, production-ready Automatic Speech Recognition (ASR) tool built using **Hugging Face Transformers (Wav2Vec 2.0)** and **PyTorch**. The tool provides both an intuitive **Web Interface** (powered by Gradio) and a flexible **Command-Line Interface (CLI)** supporting file uploads and direct microphone recording.

---

## 📌 Project Overview

This project implements modern acoustic modeling and Connectionist Temporal Classification (CTC) decoding using Meta AI's `facebook/wav2vec2-base-960h` pretrained architecture. It automatically resamples input audio into standard 16,000 Hz single-channel waveforms and maps latent acoustic representations to high-accuracy text transcriptions.

### Key Features
- **Dual Interface:** Interactive Web Application and scripted CLI.
- **Flexible Audio Input:**
  - Audio file uploads: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`.
  - Real-time microphone input with countdown and live recording.
- **Hardware Acceleration:** Native support for Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), and fallback CPU.
- **Performance Metrics:** Reports audio duration, preprocessing time, inference latency, and hardware device.
- **Export Capabilities:** Download transcripts directly from the UI or save as `.txt` / `.json` from the CLI.

---

## 🛠️ Tools & Technologies Used

| Category | Technology / Library | Description |
| :--- | :--- | :--- |
| **ASR Architecture** | [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) (`facebook/wav2vec2-base-960h`) | Self-supervised acoustic speech recognition model trained on 960 hours of LibriSpeech |
| **Deep Learning Framework** | [PyTorch](https://pytorch.org/) (`torch`, `torchaudio`) | Tensor computation and GPU/MPS hardware-accelerated inference |
| **Model Hub & NLP** | [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Model weights, tokenizer, and CTC batch decoder |
| **Audio Processing** | [Librosa](https://librosa.org/), [SoundFile](https://python-soundfile.readthedocs.io/) | High-fidelity audio loading, normalization, and sample rate resampling |
| **Microphone Input** | [SoundDevice](https://python-sounddevice.readthedocs.io/) | PortAudio bindings for live audio capture |
| **Web Interface** | [Gradio](https://www.gradio.app/) | Responsive, modern web UI with audio playback, file upload, and export |
| **Language & Environment** | Python 3.9+ | Cross-platform compatibility (macOS, Linux, Windows) |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Inputs
        A1[📁 Audio File: WAV, MP3, M4A] --> B[Pre-processing Engine]
        A2[🎙️ Live Microphone Input] --> B
    end

    subgraph Pre-processing
        B --> C[Resample to 16,000 Hz Mono]
        C --> D[Normalize Waveform [-1.0, 1.0]]
    end

    subgraph Inference Pipeline
        D --> E[Wav2Vec2Processor]
        E --> F[Wav2Vec2ForCTC Acoustic Model]
        F --> G[Logits & Argmax CTC Decoding]
    end

    subgraph Outputs
        G --> H1[🖥️ Gradio Web Application]
        G --> H2[💻 Terminal CLI / JSON]
        G --> H3[📄 Text File Export]
    end
```

---

## 📁 Project Structure

```text
.
├── asr_engine.py           # Core ASR engine with Wav2Vec2 and audio preprocessing
├── app.py                  # Gradio-based Web Application UI
├── cli.py                  # Command-Line Interface supporting files & mic
├── create_sample_audio.py  # Utility to generate synthetic test audio
├── test_asr.py             # Unit and integration test suite
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Git ignore rules for virtualenvs and temporary files
└── README.md               # Project documentation and execution guide
```

---

## 🚀 Setup & Installation

### 1. Clone or Navigate to the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: On macOS, if prompted for microphone permissions when recording, grant access in System Settings > Privacy & Security > Microphone).*

---

## 💻 How to Run

### Option 1: Launch the Web Application (Recommended)

Start the interactive web server:
```bash
python3 app.py
```
Open your browser and navigate to:
```
http://localhost:7860
```
- Select **📁 File Upload** to drag-and-drop any speech file.
- Or select **🎙️ Microphone** to record live speech.
- Click **Transcribe** to view the output and export the transcript.

---

### Option 2: Run via Command-Line Interface (CLI)

#### A. Transcribe an Audio File
```bash
python3 cli.py --file path/to/audio.wav
```

Save the transcription to a text file:
```bash
python3 cli.py --file path/to/audio.wav --output transcription.txt
```

Get detailed JSON output including duration and latency:
```bash
python3 cli.py --file path/to/audio.wav --json
```

#### B. Transcribe Live Microphone Input
Record for 5 seconds (default):
```bash
python3 cli.py --mic
```

Custom duration (e.g., 8 seconds):
```bash
python3 cli.py --mic --duration 8.0
```

#### C. Custom Model Selection
You can specify another Hugging Face Wav2Vec 2.0 checkpoint:
```bash
python3 cli.py --file sample.wav --model facebook/wav2vec2-large-960h-lv60-self-en
```

---

## 🧪 Testing the Project

Generate a synthetic test audio file and run automated test cases:

```bash
# Generate sample audio
python3 create_sample_audio.py

# Run the test suite
python3 -m unittest test_asr.py -v
```

---

## 📤 Upload to GitHub Instructions

Follow these steps to upload this project to your GitHub repository:

```bash
# 1. Initialize local Git repository (if not already initialized)
git init

# 2. Stage all project files
git add .

# 3. Create initial commit
git commit -m "feat: complete Wav2Vec2 ASR tool implementation with Web UI and CLI"

# 4. Rename branch to main
git branch -M main

# 5. Link to your GitHub repository (replace with your actual GitHub repo URL)
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 6. Push to GitHub
git push -u origin main
```

---

## 📄 License
This project is open-source under the MIT License.
