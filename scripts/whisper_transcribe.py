#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper local transcription script
- Loads local OpenAI Whisper model (supports GPU if available)
- Outputs UTF-8 JSON list of segments: [{start: float, end: float, text: str}, ...]
Usage:
    python scripts/whisper_transcribe.py <audio_file>
Env:
    WHISPER_MODEL: tiny|base|small|medium|large (default: base)
"""
import sys
import json
import os
import io

# Ensure UTF-8 stdio
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import torch
    import whisper
except Exception as e:
    print(f"Failed to import dependencies: {e}", file=sys.stderr)
    sys.exit(2)


def transcribe_audio(audio_path: str):
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    model_size = os.getenv("WHISPER_MODEL", "base")
    print(f"Loading Whisper model: {model_size}", file=sys.stderr)
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"Load model failed: {e}", file=sys.stderr)
        return None

    options = {
        "language": os.getenv("ASR_LANGUAGE", "zh"),
        "task": "transcribe",
        "fp16": torch.cuda.is_available(),
        "verbose": False,
    }
    try:
        result = model.transcribe(audio_path, **options)
    except Exception as e:
        print(f"Transcribe failed: {e}", file=sys.stderr)
        return None

    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        })

    if not segments and result.get("text"):
        segments = [{
            "start": 0.0,
            "end": float(result.get("duration", 0.0) or 0.0),
            "text": (result.get("text") or "").strip(),
        }]

    return segments


def main():
    if len(sys.argv) != 2:
        print("Usage: python whisper_transcribe.py <audio_file>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    segments = transcribe_audio(audio_path)
    if segments is None:
        sys.exit(1)

    print(json.dumps(segments, ensure_ascii=False))


if __name__ == "__main__":
    main()