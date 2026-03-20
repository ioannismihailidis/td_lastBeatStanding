"""
beat_detection_onnx — Standalone real-time beat detection using ONNX models.

Export models once from madmom, then run with only numpy + onnxruntime.

Quick start:
    # 1. Export (once, needs madmom + onnx):
    python -m beat_detection_onnx.export_models

    # 2. Use (only needs numpy + onnxruntime):
    from beat_detection_onnx import BeatDetector, PostProcessor

    detector = BeatDetector('beat_detection_onnx/models/')
    post = PostProcessor()

    result = detector.process(audio_chunk, sample_rate=44100)
    result = post.process(result, dt=1/60, audio=audio_chunk)
"""

try:
    from .beat_detector import BeatDetector
    from .post_processor import PostProcessor
    __all__ = ['BeatDetector', 'PostProcessor']
except ImportError:
    # onnxruntime not available — export_models.py can still run
    __all__ = []
