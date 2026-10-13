"""
Command-Line Interface (CLI) for Wav2Vec2 ASR Tool.
Supports audio file transcription and direct microphone recording.
"""

import os
import sys
import argparse
import json
import time

def record_microphone(duration: float = 5.0, sample_rate: int = 16000):
    """Record audio from the default microphone for a given duration."""
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] 'sounddevice' package is required for microphone recording.")
        print("Please install it using: pip install sounddevice")
        sys.exit(1)

    print(f"\n[*] Preparing microphone recording for {duration:.1f} seconds...")
    for i in range(3, 0, -1):
        print(f"[*] Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    
    print("\n[>>>] RECORDING NOW... Speak into your microphone! [<<<]")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("[✓] Recording complete!")
    return recording.flatten(), sample_rate

def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Wav2Vec 2.0 Automatic Speech Recognition (ASR) CLI Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to an audio file (WAV, MP3, M4A, FLAC, OGG) to transcribe."
    )
    group.add_argument(
        "-m", "--mic",
        action="store_true",
        help="Record live audio from microphone and transcribe."
    )

    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=5.0,
        help="Microphone recording duration in seconds (default: 5.0)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face model ID (default: facebook/wav2vec2-base-960h)."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Compute device (auto-detects CUDA/MPS/CPU if omitted)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to save transcription output (.txt or .json)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as formatted JSON to stdout."
    )

    args = parser.parse_args()

    print("=" * 60)
    print("      🎙️  WAV2VEC 2.0 AUTOMATIC SPEECH RECOGNITION (ASR)     ")
    print("=" * 60)

    from asr_engine import Wav2Vec2ASR

    # Initialize ASR
    engine = Wav2Vec2ASR(model_id=args.model, device=args.device)

    # Process input
    if args.file:
        if not os.path.isfile(args.file):
            print(f"[ERROR] Audio file not found: {args.file}")
            sys.exit(1)
        print(f"[*] Processing audio file: {args.file}")
        audio_source = args.file
    elif args.mic:
        audio_data, sr = record_microphone(duration=args.duration)
        audio_source = (sr, audio_data)

    print("[*] Transcribing speech...")
    result = engine.transcribe(audio_source)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "-" * 60)
        print("TRANSCRIPTION RESULT:")
        print("-" * 60)
        print(result["text"] if result["text"] else "[No speech detected or empty audio]")
        print("-" * 60)
        print(f"Audio Duration      : {result['duration_sec']}s")
        print(f"Inference Latency   : {result['inference_time_sec']}s")
        print(f"Compute Device      : {result['device']}")
        print(f"Model ID            : {result['model']}")
        print("-" * 60)

    # Save to output file if requested
    if args.output:
        out_ext = os.path.splitext(args.output)[1].lower()
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if out_ext == ".json":
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result["text"] + "\n")
        print(f"[✓] Saved transcription result to: {args.output}")

if __name__ == "__main__":
    main()
