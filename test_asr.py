"""
Test suite for Wav2Vec 2.0 ASR Tool.
Tests engine initialization, audio preprocessing, and end-to-end transcription pipeline.
"""

import os
import unittest
import numpy as np
from create_sample_audio import generate_tone_wav

class TestASRTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare sample audio and initialize ASR engine once for test suite."""
        cls.test_audio_path = "test_sample.wav"
        generate_tone_wav(cls.test_audio_path, duration_sec=2.0, sample_rate=16000)

    @classmethod
    def tearDownClass(cls):
        """Clean up generated test audio file."""
        if os.path.exists(cls.test_audio_path):
            os.remove(cls.test_audio_path)

    def test_audio_file_exists(self):
        """Verify that synthetic test audio file is generated properly."""
        self.assertTrue(os.path.isfile(self.test_audio_path))
        self.assertGreater(os.path.getsize(self.test_audio_path), 0)

    def test_engine_initialization(self):
        """Verify that Wav2Vec2ASR engine initializes model and processor."""
        from asr_engine import Wav2Vec2ASR
        engine = Wav2Vec2ASR()
        self.assertIsNotNone(engine.processor)
        self.assertIsNotNone(engine.model)
        self.assertIn(engine.device, ["cuda", "mps", "cpu"])

    def test_audio_preprocessing_tuple(self):
        """Verify handling of tuple (sr, data) audio input format."""
        from asr_engine import Wav2Vec2ASR
        engine = Wav2Vec2ASR()

        # Synthetic 44100Hz stereo int16 array
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        mono_signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        stereo_signal = np.stack([mono_signal, mono_signal], axis=1)

        processed = engine.load_audio((sr, stereo_signal))
        self.assertEqual(processed.ndim, 1)
        self.assertEqual(len(processed), int(engine.TARGET_SAMPLE_RATE * duration))
        self.assertTrue(np.issubdtype(processed.dtype, np.floating))

    def test_end_to_end_transcription(self):
        """Test end-to-end transcription pipeline."""
        from asr_engine import Wav2Vec2ASR
        engine = Wav2Vec2ASR()
        result = engine.transcribe(self.test_audio_path)

        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("duration_sec", result)
        self.assertIn("inference_time_sec", result)
        self.assertIn("device", result)
        self.assertIn("model", result)
        self.assertAlmostEqual(result["duration_sec"], 2.0, delta=0.2)
        self.assertGreater(result["inference_time_sec"], 0)

if __name__ == "__main__":
    unittest.main()
