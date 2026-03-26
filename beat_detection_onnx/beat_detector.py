"""
Standalone real-time beat detector using numpy-only inference.

Weights are loaded from .npz files (extracted from ONNX models exported from madmom).

Dependencies: numpy (only)
No madmom or onnxruntime dependency at runtime.
"""

import io
import os
import json
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Preprocessing: Audio → Feature frames
# ─────────────────────────────────────────────────────────────────────

class Preprocessor:
    """Audio feature extraction matching madmom's RNNBeatProcessor pipeline.

    Pipeline: Audio → Frame → STFT → Filterbank → Log → Diff → Stack
    """

    def __init__(self, filterbank, frame_size=2048, hop_size=441,
                 log_add=1.0, log_mul=1.0):
        self.filterbank = filterbank          # (num_fft_bins, num_bands)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.log_add = log_add
        self.log_mul = log_mul

        self.window = np.hanning(frame_size).astype(np.float32)
        self.prev_filtered = None             # for spectral difference

    def reset(self):
        self.prev_filtered = None

    def process_frames(self, audio_frames):
        """Process framed audio into feature vectors.

        Args:
            audio_frames: numpy array (num_frames, frame_size)

        Returns:
            features: numpy array (num_frames, num_features)
        """
        num_frames = audio_frames.shape[0]
        num_bands = self.filterbank.shape[1]
        features = np.zeros((num_frames, num_bands * 2), dtype=np.float32)

        for i in range(num_frames):
            frame = audio_frames[i]

            # 1. Window
            windowed = frame * self.window

            # 2. FFT → magnitude spectrum
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft).astype(np.float32)

            # 3. Apply filterbank
            filtered = magnitude @ self.filterbank

            # 4. Logarithmic compression: log10(x * mul + add)
            log_filtered = np.log10(filtered * self.log_mul + self.log_add)

            # 5. Spectral difference (positive only)
            if self.prev_filtered is not None:
                diff = log_filtered - self.prev_filtered
                diff = np.maximum(0, diff)
            else:
                diff = np.zeros(num_bands, dtype=np.float32)
            self.prev_filtered = log_filtered.copy()

            # 6. Stack: [log_spec, diff]
            features[i, :num_bands] = log_filtered
            features[i, num_bands:] = diff

        return features


def build_log_filterbank(num_bands, fmin, fmax, num_fft_bins, sample_rate):
    """Build a logarithmic filterbank matrix.

    Fallback if the exported filterbank.npy is not available.
    Note: May not match madmom's filterbank exactly.
    """
    fft_freqs = np.linspace(0, sample_rate / 2, num_fft_bins)

    log_fmin = np.log2(max(fmin, 1))
    log_fmax = np.log2(fmax)
    center_freqs = 2 ** np.linspace(log_fmin, log_fmax, num_bands + 2)

    filterbank = np.zeros((num_fft_bins, num_bands), dtype=np.float32)
    for i in range(num_bands):
        f_low = center_freqs[i]
        f_center = center_freqs[i + 1]
        f_high = center_freqs[i + 2]

        mask = (fft_freqs >= f_low) & (fft_freqs <= f_center)
        if f_center > f_low:
            filterbank[mask, i] = (fft_freqs[mask] - f_low) / (f_center - f_low)

        mask = (fft_freqs > f_center) & (fft_freqs <= f_high)
        if f_high > f_center:
            filterbank[mask, i] = (f_high - fft_freqs[mask]) / (f_high - f_center)

    col_sums = filterbank.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    filterbank /= col_sums

    return filterbank


# ─────────────────────────────────────────────────────────────────────
# Pure Numpy LSTM Inference
# ─────────────────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _tanh(x):
    return np.tanh(x)


class NumpyLSTMModel:
    """Single LSTM model with 3 LSTM layers + dense output, pure numpy.

    Weights are stored in ONNX gate order: i, o, f, c
    W shape: (1, 4*H, input_size)  — input weights
    R shape: (1, 4*H, H)           — recurrent weights
    B shape: (1, 8*H)              — biases (first 4*H input, next 4*H recurrent)

    Args:
        source: str (file path) or bytes/BytesIO (in-memory npz data)
    """

    def __init__(self, source):
        if isinstance(source, (bytes, bytearray, memoryview)):
            source = io.BytesIO(bytes(source))
        data = np.load(source)

        self.num_layers = 0
        self.layers = []

        # Load LSTM layers
        while f'W_{self.num_layers}' in data:
            i = self.num_layers
            H = data[f'R_{i}'].shape[2]

            layer = {
                'W': data[f'W_{i}'][0],       # (4*H, input_size)
                'R': data[f'R_{i}'][0],        # (4*H, H)
                'B': data[f'B_{i}'][0],        # (8*H,)
                'H': H,
            }
            self.layers.append(layer)
            self.num_layers += 1

        # Dense output layer
        self.dense_W = data['dense_W']   # (H, 1)
        self.dense_b = data['dense_b']   # (1,)

        # LSTM states: h and c for each layer
        self.h = [np.zeros(l['H'], dtype=np.float32) for l in self.layers]
        self.c = [np.zeros(l['H'], dtype=np.float32) for l in self.layers]

        # Load initial states if available
        for i in range(self.num_layers):
            key_h = f'h_{i}_init_default'
            key_c = f'c_{i}_init_default'
            if key_h in data:
                self.h[i] = data[key_h].flatten().astype(np.float32).copy()
            if key_c in data:
                self.c[i] = data[key_c].flatten().astype(np.float32).copy()

        # Store initial states for reset
        self._h_init = [h.copy() for h in self.h]
        self._c_init = [c.copy() for c in self.c]

    def reset(self):
        for i in range(self.num_layers):
            self.h[i] = self._h_init[i].copy()
            self.c[i] = self._c_init[i].copy()

    def process(self, features):
        """Run features through LSTM layers + dense output.

        Args:
            features: numpy array (num_frames, num_features)

        Returns:
            activations: numpy array (num_frames,)
        """
        num_frames = features.shape[0]
        x = features  # (num_frames, input_size)

        # Process through each LSTM layer
        for li, layer in enumerate(self.layers):
            W = layer['W']    # (4*H, input_size)
            R = layer['R']    # (4*H, H)
            B = layer['B']    # (8*H,)
            H = layer['H']

            # Split biases: input bias + recurrent bias
            Wb = B[:4*H]
            Rb = B[4*H:8*H]

            h = self.h[li]
            c = self.c[li]

            output = np.zeros((num_frames, H), dtype=np.float32)

            for t in range(num_frames):
                xt = x[t]

                # gates = W @ xt + R @ h + Wb + Rb
                # ONNX gate order: i, o, f, c
                gates = W @ xt + R @ h + Wb + Rb

                i_gate = _sigmoid(gates[0*H:1*H])
                o_gate = _sigmoid(gates[1*H:2*H])
                f_gate = _sigmoid(gates[2*H:3*H])
                c_gate = _tanh(gates[3*H:4*H])

                c = f_gate * c + i_gate * c_gate
                h = o_gate * _tanh(c)

                output[t] = h

            self.h[li] = h
            self.c[li] = c
            x = output  # feed to next layer

        # Dense + sigmoid
        logits = x @ self.dense_W + self.dense_b  # (num_frames, 1)
        activations = _sigmoid(logits.flatten())

        return activations


class LSTMEnsemble:
    """Manages multiple numpy LSTM models for ensemble beat detection.

    Args:
        model_dir: str path to directory with beat_lstm_*.npz files, OR
                   None if model_files is provided.
        model_files: dict mapping filenames to bytes/BytesIO (e.g. from VFS).
                     Keys should be like 'beat_lstm_1.npz', etc.
        num_models: Limit to first N models.
        single_model: Use only 1 model (fastest).
    """

    def __init__(self, model_dir=None, model_files=None,
                 num_models=None, single_model=False):
        if model_files is not None:
            # Load from in-memory data (VFS, etc.)
            npz_keys = sorted([
                k for k in model_files
                if k.endswith('.npz') and k.startswith('beat_lstm_')
            ])
            if not npz_keys:
                raise FileNotFoundError("No beat_lstm_*.npz entries in model_files")
            if single_model:
                npz_keys = npz_keys[:1]
            elif num_models is not None:
                npz_keys = npz_keys[:num_models]
            sources = [model_files[k] for k in npz_keys]
        else:
            # Load from filesystem
            npz_files = sorted([
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith('.npz') and f.startswith('beat_lstm_')
            ])
            if not npz_files:
                raise FileNotFoundError(
                    f"No .npz model weights found in {model_dir}")
            if single_model:
                npz_files = npz_files[:1]
            elif num_models is not None:
                npz_files = npz_files[:num_models]
            sources = npz_files

        self.models = [NumpyLSTMModel(s) for s in sources]
        self.num_models = len(self.models)

    def reset(self):
        for model in self.models:
            model.reset()

    def process(self, features):
        """Run features through all models and average activations.

        Args:
            features: numpy array (num_frames, num_features)

        Returns:
            activations: numpy array (num_frames,) — beat activation [0, 1]
        """
        all_activations = [m.process(features) for m in self.models]
        return np.mean(all_activations, axis=0)


# ─────────────────────────────────────────────────────────────────────
# DBN Beat Tracker (online forward algorithm)
# ─────────────────────────────────────────────────────────────────────

class DBNBeatTracker:
    """Dynamic Bayesian Network beat tracker using online forward algorithm.

    Reimplements madmom's DBNBeatTrackingProcessor in pure numpy.
    Uses a tempo-phase state space with Viterbi-like forward decoding.
    """

    def __init__(self, min_bpm=55, max_bpm=215, fps=100,
                 transition_lambda=100, observation_lambda=16):
        self.fps = fps
        self.observation_lambda = observation_lambda
        self.transition_lambda = transition_lambda

        # Beat intervals in frames
        min_interval = max(1, int(np.round(60.0 * fps / max_bpm)))
        max_interval = int(np.round(60.0 * fps / min_bpm))

        # Build state space: each state = (tempo_index, phase)
        intervals = list(range(min_interval, max_interval + 1))
        self.unique_intervals = np.array(intervals, dtype=np.int32)
        self.num_tempos = len(intervals)

        # State arrays
        positions = []
        state_intervals = []

        self.beat_state_indices = []
        self.boundary_indices = []
        state_offset = 0

        for iv in intervals:
            self.beat_state_indices.append(state_offset)
            for phase in range(iv):
                positions.append(phase / iv)
                state_intervals.append(iv)
            self.boundary_indices.append(state_offset + iv - 1)
            state_offset += iv

        self.positions = np.array(positions, dtype=np.float32)
        self.state_intervals = np.array(state_intervals, dtype=np.int32)
        self.num_states = len(positions)

        self.beat_state_indices = np.array(self.beat_state_indices, dtype=np.int32)
        self.boundary_indices = np.array(self.boundary_indices, dtype=np.int32)

        # Observation model
        border = 1.0 / observation_lambda
        self.pointers = (self.positions < border).astype(np.int32)

        # Precompute transition structures
        self._build_transitions()

        # Forward state (log probabilities)
        self.fwd = np.zeros(self.num_states, dtype=np.float64)
        self.fwd[:] = -np.log(self.num_states)

        self._frame_counter = 0
        self._last_beat_frame = -1

    def _build_transitions(self):
        boundary_set = set(self.boundary_indices.tolist())

        self.advance_from = np.array(
            [i for i in range(self.num_states) if i not in boundary_set],
            dtype=np.int32
        )
        self.advance_to = self.advance_from + 1

        src_ivs = self.unique_intervals.astype(np.float64)[:, None]
        tgt_ivs = self.unique_intervals.astype(np.float64)[None, :]
        ratios = tgt_ivs / src_ivs
        self.boundary_trans_probs = np.exp(
            -self.transition_lambda * np.abs(ratios - 1.0)
        )
        row_sums = self.boundary_trans_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.boundary_trans_probs /= row_sums

        with np.errstate(divide='ignore'):
            self.log_boundary_trans = np.log(self.boundary_trans_probs)

    def reset(self):
        self.fwd[:] = -np.log(self.num_states)
        self._frame_counter = 0
        self._last_beat_frame = -1

    def process_frame(self, activation):
        """Process a single activation frame.

        Args:
            activation: float, beat activation value [0, 1]

        Returns:
            is_beat: bool, True if a beat is detected at this frame
        """
        activation = float(np.clip(activation, 1e-7, 1.0 - 1e-7))

        obs_lambda = self.observation_lambda
        log_beat = np.log(activation)
        log_no_beat = np.log((1.0 - activation) / max(obs_lambda - 1, 1))

        log_obs = np.where(self.pointers == 1, log_beat, log_no_beat)

        new_fwd = np.full(self.num_states, -np.inf, dtype=np.float64)

        # Phase advance
        new_fwd[self.advance_to] = self.fwd[self.advance_from]

        # Beat boundary transitions
        src_probs = self.fwd[self.boundary_indices]
        contrib = src_probs[:, None] + self.log_boundary_trans

        max_contrib = contrib.max(axis=0)
        valid = max_contrib > -np.inf
        beat_fwd = np.full(self.num_tempos, -np.inf, dtype=np.float64)
        if valid.any():
            safe = contrib[:, valid] - max_contrib[valid]
            beat_fwd[valid] = max_contrib[valid] + np.log(np.exp(safe).sum(axis=0))

        # Merge
        beat_targets = self.beat_state_indices
        existing = new_fwd[beat_targets]
        stacked = np.stack([existing, beat_fwd])
        max_stack = stacked.max(axis=0)
        valid2 = max_stack > -np.inf
        if valid2.any():
            safe2 = stacked[:, valid2] - max_stack[valid2]
            new_fwd[beat_targets[valid2]] = max_stack[valid2] + np.log(
                np.exp(safe2).sum(axis=0)
            )

        # Apply observation
        new_fwd += log_obs

        # Normalize
        max_fwd = new_fwd.max()
        if max_fwd > -np.inf:
            log_sum = max_fwd + np.log(np.exp(new_fwd - max_fwd).sum())
            new_fwd -= log_sum

        self.fwd = new_fwd
        self._frame_counter += 1

        # Beat detection
        best_state = np.argmax(self.fwd)
        is_beat = bool(self.pointers[best_state] == 1)

        if is_beat:
            self._last_beat_frame = self._frame_counter

        return is_beat


# ─────────────────────────────────────────────────────────────────────
# Main BeatDetector class
# ─────────────────────────────────────────────────────────────────────

class BeatDetector:
    """Real-time beat detector using pure numpy inference.

    Combines audio preprocessing, LSTM neural network inference,
    and Dynamic Bayesian Network beat tracking.

    Usage (filesystem):
        detector = BeatDetector(model_dir='path/to/models/')

    Usage (TouchDesigner VFS):
        vfs_files = {f.name: bytes(f.byteArray) for f in me.parent().vfs}
        detector = BeatDetector(model_files=vfs_files)
    """

    BPM_HISTORY = 8
    MIN_DBN_FRAMES = 4
    TARGET_SR = 44100
    FPS = 100

    def __init__(self, model_dir=None, model_files=None,
                 min_bpm=60, max_bpm=190,
                 transition_lambda=100, observation_lambda=16,
                 act_gate=0.15, rms_gate=0.005, single_model=True):
        """
        Args:
            model_dir: Path to directory with model files (filesystem).
            model_files: Dict mapping filenames to bytes (in-memory / VFS).
                         Expected keys: 'config.json', 'filterbank.npy',
                         'beat_lstm_1.npz', etc.
                         Provide either model_dir or model_files, not both.
        """
        if model_dir is None and model_files is None:
            raise ValueError("Provide either model_dir or model_files")

        # --- Load config ---
        if model_files is not None and 'config.json' in model_files:
            raw = model_files['config.json']
            if isinstance(raw, (bytes, bytearray, memoryview)):
                raw = bytes(raw).decode('utf-8')
            self.config = json.loads(raw)
        else:
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        # --- Load filterbank ---
        if model_files is not None and 'filterbank.npy' in model_files:
            fb_data = model_files['filterbank.npy']
            if isinstance(fb_data, (bytes, bytearray, memoryview)):
                fb_data = io.BytesIO(bytes(fb_data))
            filterbank = np.load(fb_data)
        elif model_dir is not None:
            fb_path = os.path.join(model_dir, 'filterbank.npy')
            if os.path.exists(fb_path):
                filterbank = np.load(fb_path)
            else:
                filterbank = None
        else:
            filterbank = None

        if filterbank is None:
            num_fft_bins = self.config['frame_size'] // 2 + 1
            filterbank = build_log_filterbank(
                num_bands=self.config['num_filter_bands'],
                fmin=self.config['fmin'],
                fmax=self.config['fmax'],
                num_fft_bins=num_fft_bins,
                sample_rate=self.config['sample_rate'],
            )

        # Initialize components
        self._prep = Preprocessor(
            filterbank=filterbank,
            frame_size=self.config['frame_size'],
            hop_size=self.config['hop_size'],
            log_add=self.config.get('log_add', 1.0),
            log_mul=self.config.get('log_mul', 1.0),
        )

        self._ensemble = LSTMEnsemble(
            model_dir=model_dir,
            model_files=model_files,
            single_model=single_model,
        )

        self._dbn = DBNBeatTracker(
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=self.FPS,
            transition_lambda=transition_lambda,
            observation_lambda=observation_lambda,
        )

        # Parameters
        self.act_gate = act_gate
        self.rms_gate = rms_gate
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

        # State
        self._audio_buf = np.array([], dtype=np.float32)
        self._act_buffer = []
        self._sample_count = 0
        self._last_beat_t = -1.0
        self._prev_beat_t = -1.0
        self._beat_intervals = []
        self._bpm = 0.0
        self._last_interval = 0.0
        self._sr = self.TARGET_SR

        self._hop_size = self.config['hop_size']
        self._frame_size = self.config['frame_size']

    def reset(self):
        """Reset all state for a fresh start."""
        self._prep.reset()
        self._ensemble.reset()
        self._dbn.reset()
        self._audio_buf = np.array([], dtype=np.float32)
        self._act_buffer = []
        self._sample_count = 0
        self._last_beat_t = -1.0
        self._prev_beat_t = -1.0
        self._beat_intervals = []
        self._bpm = 0.0
        self._last_interval = 0.0

    def process(self, audio, sample_rate=44100):
        """Process an audio chunk and detect beats.

        Call this repeatedly with consecutive audio chunks.

        Args:
            audio: numpy array (num_samples,) — mono float32 audio
            sample_rate: Input sample rate (will resample to 44100 if needed)

        Returns:
            dict with keys:
                beat (bool): True if a beat was detected in this chunk
                bpm (float): Current estimated BPM
                beat_interval (float): Time between last two beats in seconds
                confidence (float): Peak activation value
                phase (float): Current position within beat cycle [0, 1)
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        self._sr = sample_rate

        # Resample to target SR if needed
        if sample_rate != self.TARGET_SR:
            n_target = int(round(len(audio) * self.TARGET_SR / sample_rate))
            if n_target > 0 and len(audio) > 1:
                x_old = np.linspace(0, 1, len(audio))
                x_new = np.linspace(0, 1, n_target)
                audio = np.interp(x_new, x_old, audio).astype(np.float32)

        self._sample_count += len(audio)

        # Accumulate in frame-aligned buffer
        self._audio_buf = np.concatenate([self._audio_buf, audio])

        available = len(self._audio_buf)
        if available < self._frame_size:
            return self._make_result(beat=False, confidence=0.0)

        n_frames = 1 + (available - self._frame_size) // self._hop_size
        if n_frames <= 0:
            return self._make_result(beat=False, confidence=0.0)

        # Extract frames
        frames = np.zeros((n_frames, self._frame_size), dtype=np.float32)
        for i in range(n_frames):
            start = i * self._hop_size
            frames[i] = self._audio_buf[start:start + self._frame_size]

        # Advance buffer
        consumed = n_frames * self._hop_size
        self._audio_buf = self._audio_buf[consumed:]

        # Preprocess: frames → features
        features = self._prep.process_frames(frames)

        # LSTM inference: features → activations
        activations = self._ensemble.process(features)

        if activations.size == 0:
            return self._make_result(beat=False, confidence=0.0)

        # Confidence + RMS gating
        peak_act = float(activations.max())
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
        gated = rms < self.rms_gate or peak_act < self.act_gate

        # Buffer activations for DBN
        self._act_buffer.append(activations)
        total_frames = sum(a.size for a in self._act_buffer)

        if total_frames < self.MIN_DBN_FRAMES:
            return self._make_result(beat=False, confidence=peak_act)

        # Run DBN on buffered activations
        batched_acts = np.concatenate(self._act_buffer)
        self._act_buffer = []

        beat_detected = False
        current_t = self._sample_count / float(self._sr)

        if not gated:
            for act in batched_acts:
                is_beat = self._dbn.process_frame(float(act))
                if is_beat:
                    frame_t = current_t
                    if frame_t > self._last_beat_t:
                        self._last_beat_t = frame_t

                        if self._prev_beat_t > 0:
                            interval = frame_t - self._prev_beat_t
                            if interval > 0:
                                self._last_interval = round(interval, 4)
                                self._update_bpm(interval)

                        self._prev_beat_t = frame_t
                        beat_detected = True
        else:
            for act in batched_acts:
                self._dbn.process_frame(float(act))

        return self._make_result(beat=beat_detected, confidence=peak_act)

    def _update_bpm(self, interval):
        inst_bpm = 60.0 / interval
        if inst_bpm < self.min_bpm or inst_bpm > self.max_bpm:
            return

        self._beat_intervals.append(interval)
        if len(self._beat_intervals) > self.BPM_HISTORY:
            self._beat_intervals = self._beat_intervals[-self.BPM_HISTORY:]

        intervals = sorted(self._beat_intervals)
        median_iv = intervals[len(intervals) // 2]
        self._bpm = round(60.0 / median_iv, 2)

    def _make_result(self, beat, confidence):
        phase = 0.0
        if self._bpm > 0 and self._prev_beat_t > 0:
            current_t = self._sample_count / float(self._sr)
            expected_iv = 60.0 / self._bpm
            if expected_iv > 0:
                phase = (current_t - self._prev_beat_t) / expected_iv % 1.0

        return {
            'beat': beat,
            'bpm': self._bpm,
            'beat_interval': self._last_interval,
            'confidence': confidence,
            'phase': phase,
        }

    @property
    def bpm(self):
        return self._bpm

    @property
    def beat_interval(self):
        return self._last_interval
