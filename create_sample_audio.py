"""
Utility to generate synthetic sample WAV audio files for testing the ASR tool.
Uses Python standard library (wave, struct, math) so no extra dependencies are needed.
"""

import math
import wave
import struct
import os

def generate_tone_wav(file_path: str = "sample_audio.wav", duration_sec: float = 3.0, sample_rate: int = 16000, freq: float = 440.0):
    """
    Generate a simple sine-wave tone audio file in 16-bit PCM WAV format.
    """
    num_samples = int(duration_sec * sample_rate)
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with wave.open(file_path, "w") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit (2 bytes)
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            # Generate modulated sine tone
            sample_val = 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate))
            # Quantize to 16-bit signed integer
            packed_val = struct.pack("<h", int(sample_val * 32767.0))
            wav_file.writeframesraw(packed_val)

    print(f"[✓] Generated synthetic sample audio: {file_path} ({duration_sec}s, {sample_rate}Hz mono)")
    return file_path

if __name__ == "__main__":
    generate_tone_wav("sample_audio.wav", duration_sec=3.0)
