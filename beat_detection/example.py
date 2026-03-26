#!/usr/bin/env python
"""
Example: Real-time beat detection from an audio file.

Usage:
    1. First, export models (once):
       python -m beat_detection.export_models

    2. Run this example:
       python -m beat_detection.example path/to/audio.wav

Requirements:
    pip install numpy soundfile
"""

import sys
import os
import time
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m beat_detection.example <audio_file>")
        print("\nSupported formats: WAV, FLAC, OGG (requires soundfile)")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    if not os.path.exists(os.path.join(model_dir, 'config.json')):
        print("ERROR: Models not found. Run export first:")
        print("  python -m beat_detection.export_models")
        sys.exit(1)

    # Load audio
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype='float32')
    except ImportError:
        print("Install soundfile for audio loading: pip install soundfile")
        print("Alternatively, provide raw numpy audio data.")
        sys.exit(1)

    # Mono downmix
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    print(f"Audio: {audio_path}")
    print(f"  Duration: {len(audio)/sr:.1f}s, Sample rate: {sr} Hz")
    print()

    # Initialize detector
    from beat_detection import BeatDetector, PostProcessor

    detector = BeatDetector(
        model_dir,
        min_bpm=60,
        max_bpm=190,
        single_model=True,
    )
    post = PostProcessor(
        time_mult=1.0,
        beats_per_bar=4,
    )

    # Process in chunks (simulating real-time)
    chunk_size = int(sr / 100)  # ~10ms chunks (100 FPS)
    total_chunks = len(audio) // chunk_size
    beat_count = 0
    start_time = time.time()

    print("Processing...")
    print("-" * 60)

    for i in range(total_chunks):
        chunk = audio[i * chunk_size:(i + 1) * chunk_size]
        dt = chunk_size / sr

        result = detector.process(chunk, sample_rate=sr)
        result = post.process(result, dt=dt, audio=chunk, sample_rate=sr)

        if result['beat']:
            beat_count += 1
            t = (i * chunk_size) / sr
            synth_tag = " [synth]" if result['synth'] else ""
            print(
                f"  Beat #{beat_count:3d} at {t:6.2f}s | "
                f"BPM: {result['bpm']:6.1f} | "
                f"Beat {result['beat_num']}/{post.beats_per_bar} | "
                f"Conf: {result['confidence']:.3f}"
                f"{synth_tag}"
            )

    elapsed = time.time() - start_time
    audio_duration = len(audio) / sr
    print("-" * 60)
    print(f"Total beats: {beat_count}")
    print(f"Final BPM: {detector.bpm:.1f}")
    print(f"Processing time: {elapsed:.2f}s ({audio_duration/elapsed:.1f}x realtime)")


if __name__ == '__main__':
    main()
