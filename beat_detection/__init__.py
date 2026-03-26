"""
beat_detection — Standalone real-time beat detection (numpy only).

Model weights exported from madmom LSTM models. At runtime, only numpy is needed.

Quick start:
    # 1. Export (once, needs madmom + onnx):
    python -m beat_detection.export_models

    # 2. Use (only needs numpy):
    from beat_detection import BeatDetector, PostProcessor

    detector = BeatDetector('beat_detection/models/')
    post = PostProcessor()

    result = detector.process(audio_chunk, sample_rate=44100)
    result = post.process(result, dt=1/60, audio=audio_chunk)
"""

from .beat_detector import BeatDetector
from .post_processor import PostProcessor

__all__ = ['BeatDetector', 'PostProcessor']
