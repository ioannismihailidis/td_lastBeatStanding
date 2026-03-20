"""
Post-processing for beat detection output.

Provides synthetic beat continuation, beat counting,
bass energy analysis, and breakdown detection.

No TouchDesigner dependency — pure Python/numpy.
"""

import numpy as np


# Bass isolation: block-average cutoff frequency in Hz
BASS_CUTOFF = 150
# Rolling audio buffer window for bass analysis (seconds)
BASS_WINDOW = 0.2


class PostProcessor:
    """Post-processes beat detection results.

    Features:
    - Synthetic beat generation during silence (keeps the beat going)
    - Time multiplier (0.5x, 1x, 2x, 4x)
    - Beat counter with configurable beats per bar
    - Bass energy analysis for breakdown detection

    Usage:
        post = PostProcessor()
        result = post.process(
            beat_result=detector.process(audio_chunk),
            dt=1/60,  # time since last call
            audio=audio_chunk,
            sample_rate=44100,
        )
    """

    def __init__(self, time_mult=1.0, beats_per_bar=4,
                 max_synth_beats=16, bass_thresh=0.005,
                 bass_smooth=0.15):
        """
        Args:
            time_mult: BPM multiplier (0.5, 1.0, 2.0, 4.0)
            beats_per_bar: Number of beats per bar for beat_num counter
            max_synth_beats: Maximum synthetic beats before stopping
            bass_thresh: Bass energy threshold for breakdown detection
            bass_smooth: Bass energy smoothing factor (0-1)
        """
        self.time_mult = time_mult
        self.beats_per_bar = beats_per_bar
        self.max_synth_beats = max_synth_beats
        self.bass_thresh = bass_thresh
        self.bass_smooth = bass_smooth

        # State
        self._running_time = 0.0
        self._last_beat_time = -1.0
        self._last_real_beat_time = -1.0
        self._synth_beat_time = -1.0
        self._last_accepted_time = -1.0
        self._beat_counter = 0
        self._prev_in_beat = False
        self._bass_energy = 0.0
        self._audio_buf = np.array([], dtype=np.float32)
        self._audio_sr = 44100

    def reset(self):
        """Reset all state."""
        self._running_time = 0.0
        self._last_beat_time = -1.0
        self._last_real_beat_time = -1.0
        self._synth_beat_time = -1.0
        self._last_accepted_time = -1.0
        self._beat_counter = 0
        self._prev_in_beat = False
        self._bass_energy = 0.0
        self._audio_buf = np.array([], dtype=np.float32)

    def reset_beat_counter(self):
        """Reset beat counter to 0."""
        self._beat_counter = 0

    def process(self, beat_result, dt, audio=None, sample_rate=44100):
        """Process a beat detection result with post-processing.

        Args:
            beat_result: dict from BeatDetector.process() with keys:
                beat (bool), bpm (float), confidence (float), etc.
            dt: Time elapsed since last call in seconds
            audio: Optional audio chunk for bass energy analysis
            sample_rate: Audio sample rate

        Returns:
            dict with keys:
                beat (bool): Beat detected (real or synthetic)
                bpm (float): Effective BPM (with time multiplier applied)
                beat_interval (float): Effective beat interval in seconds
                confidence (float): From input
                phase (float): Position within beat cycle [0, 1)
                synth (bool): True if this is a synthetic (continued) beat
                beat_num (int): Beat position within bar (1-based)
                breakdown (bool): True if bass energy is below threshold
                bass_energy (float): Smoothed bass energy value
        """
        self._running_time += dt
        now = self._running_time

        in_beat = beat_result.get('beat', False)
        in_bpm = beat_result.get('bpm', 0.0)
        in_confidence = beat_result.get('confidence', 0.0)

        # Apply time multiplier
        output_bpm = in_bpm * self.time_mult
        effective_iv = 60.0 / output_bpm if output_bpm > 0 else 0.0

        # --- Buffer audio for bass analysis ---
        if audio is not None:
            mono = np.asarray(audio, dtype=np.float32)
            if mono.ndim > 1:
                mono = np.mean(mono, axis=0)

            self._audio_sr = sample_rate
            self._audio_buf = np.concatenate([self._audio_buf, mono])
            max_samples = int(sample_rate * BASS_WINDOW)
            if len(self._audio_buf) > max_samples:
                self._audio_buf = self._audio_buf[-max_samples:]

        # --- Beat detection (rising edge for real beats) ---
        beat_out = False
        is_synth = False
        beat_rising = in_beat and not self._prev_in_beat
        self._prev_in_beat = in_beat

        if beat_rising:
            # Real beat from DNN
            self._last_real_beat_time = now
            self._last_beat_time = now
            self._synth_beat_time = now
            beat_out = True
        elif effective_iv > 0 and self._synth_beat_time > 0 and not in_beat:
            # Check if we should generate a synthetic beat
            time_since_last = now - self._synth_beat_time
            if time_since_last >= effective_iv * 0.95:
                # Check max silence limit
                silence_ok = True
                if self.max_synth_beats > 0 and self._last_real_beat_time > 0:
                    silence_duration = now - self._last_real_beat_time
                    silence_count = int(silence_duration / effective_iv)
                    if silence_count >= self.max_synth_beats:
                        silence_ok = False

                if silence_ok:
                    self._synth_beat_time = now
                    self._last_beat_time = now
                    beat_out = True
                    is_synth = True

        # --- Bass energy analysis at every beat ---
        if beat_out:
            buf = self._audio_buf
            if len(buf) > 0:
                sr = self._audio_sr
                block_size = max(1, sr // BASS_CUTOFF)
                n_full = (len(buf) // block_size) * block_size

                if n_full >= block_size:
                    blocks = buf[:n_full].reshape(-1, block_size)
                    bass_rms = float(np.sqrt(np.mean(blocks.mean(axis=1) ** 2)))
                else:
                    bass_rms = float(np.sqrt(np.mean(buf ** 2)))

                alpha = max(0.01, min(1.0, self.bass_smooth))
                self._bass_energy = (
                    self._bass_energy * (1.0 - alpha) + bass_rms * alpha
                )

        # --- Debounce: suppress double-fires ---
        if beat_out and effective_iv > 0 and self._last_accepted_time > 0:
            if (now - self._last_accepted_time) < effective_iv * 0.5:
                beat_out = False
                is_synth = False

        # --- Advance beat counter ---
        if beat_out:
            self._last_accepted_time = now
            self._beat_counter = (self._beat_counter % self.beats_per_bar) + 1

        # --- Phase ---
        phase_out = 0.0
        if effective_iv > 0 and self._last_beat_time > 0:
            phase_out = (now - self._last_beat_time) / effective_iv % 1.0

        # --- Breakdown ---
        is_breakdown = (
            self._bass_energy < self.bass_thresh
            if self._bass_energy > 0 else False
        )

        return {
            'beat': beat_out,
            'bpm': output_bpm,
            'beat_interval': effective_iv,
            'confidence': in_confidence,
            'phase': phase_out,
            'synth': is_synth,
            'beat_num': self._beat_counter,
            'breakdown': is_breakdown,
            'bass_energy': self._bass_energy,
        }

    @property
    def beat_counter(self):
        return self._beat_counter

    @property
    def bass_energy(self):
        return self._bass_energy
