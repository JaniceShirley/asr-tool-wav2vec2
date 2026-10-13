"""
Automatic Speech Recognition (ASR) Engine using Hugging Face Transformers (Wav2Vec 2.0).
Provides unified audio loading, resampling to 16kHz, and transcription.
"""

import os
import time
import logging
from typing import Union, Dict, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Wav2Vec2ASR:
    """
    ASR Engine powered by Wav2Vec2 CTC models from Hugging Face.
    """
    DEFAULT_MODEL = "facebook/wav2vec2-base-960h"
    TARGET_SAMPLE_RATE = 16000

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the Wav2Vec2 ASR engine.

        Args:
            model_id: Hugging Face model repository ID.
            device: 'cuda', 'mps' (for Apple Silicon), 'cpu', or None for auto-detection.
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self.device = self._resolve_device(device)
        self.processor = None
        self.model = None
        self._load_model()

    def _resolve_device(self, preferred_device: Optional[str]) -> str:
        """Resolve optimal compute device (CUDA > MPS > CPU)."""
        import torch
        if preferred_device:
            return preferred_device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load Wav2Vec2 processor and model from Hugging Face."""
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        logger.info(f"Loading ASR model: {self.model_id} on device: {self.device}...")
        start_time = time.time()
        
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        
        load_duration = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_duration:.2f}s.")

    def load_audio(self, audio_input: Union[str, np.ndarray, tuple]) -> np.ndarray:
        """
        Load and normalize audio into 1D float32 array resampled to 16,000 Hz.

        Args:
            audio_input:
                - File path string (WAV, MP3, FLAC, M4A, etc.)
                - Or a tuple (sample_rate, numpy_array) from Gradio/SoundDevice
                - Or 1D numpy array (assumed 16kHz)
        Returns:
            np.ndarray: 1D audio waveform at 16kHz.
        """
        import librosa

        # Case 1: Audio passed as (sample_rate, data_array)
        if isinstance(audio_input, tuple):
            sr, data = audio_input
            if data is None or len(data) == 0:
                raise ValueError("Empty audio input received.")
            data = np.asarray(data)
            # If stereo / multichannel, convert to mono
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            # Normalize int types to float [-1.0, 1.0]
            if np.issubdtype(data.dtype, np.integer):
                max_val = float(np.iinfo(data.dtype).max)
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
            # Resample if sample rate differs from 16kHz
            if sr != self.TARGET_SAMPLE_RATE:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.TARGET_SAMPLE_RATE)
            return data

        # Case 2: Audio passed as string file path
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            # librosa loads and automatically resamples to 16kHz mono
            speech, _ = librosa.load(audio_input, sr=self.TARGET_SAMPLE_RATE, mono=True)
            return speech.astype(np.float32)

        # Case 3: Raw numpy array (assumed 16kHz)
        if isinstance(audio_input, np.ndarray):
            data = audio_input
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32)

        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    def transcribe(self, audio_input: Union[str, np.ndarray, tuple]) -> Dict[str, Any]:
        """
        Transcribe audio input into text.

        Args:
            audio_input: File path, tuple (sr, data), or 1D numpy array.

        Returns:
            dict containing:
                - 'text': Transcribed text in uppercase/standard format
                - 'duration_sec': Duration of input audio in seconds
                - 'inference_time_sec': Time taken by the neural network inference
                - 'device': Device used
                - 'model': Model ID
        """
        import torch

        # Preprocess audio
        prep_start = time.time()
        speech = self.load_audio(audio_input)
        prep_time = time.time() - prep_start

        duration_sec = len(speech) / self.TARGET_SAMPLE_RATE
        if duration_sec <= 0:
            return {
                "text": "",
                "duration_sec": 0.0,
                "inference_time_sec": 0.0,
                "device": self.device,
                "model": self.model_id
            }

        # Tokenize and create input tensor
        inputs = self.processor(
            speech,
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        # Perform CTC inference
        inf_start = time.time()
        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        inference_time = time.time() - inf_start

        # Decode tokens to string
        transcription = self.processor.batch_decode(predicted_ids)[0]
        formatted_text = transcription.strip()

        return {
            "text": formatted_text,
            "duration_sec": round(duration_sec, 2),
            "inference_time_sec": round(inference_time, 3),
            "preprocessing_time_sec": round(prep_time, 3),
            "device": self.device,
            "model": self.model_id
        }
