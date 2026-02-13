# script1_callbacks DAT (Script CHOP)

import os, sys
import numpy as np
from scipy.signal import resample

# ----------------- CONDA / DLL PATH SETUP -----------------
CONDA_ENV = r"C:\Users\janni\Desktop\td_lastBeatStanding\Miniconda3\envs\td_madmom_311"
os.add_dll_directory(os.path.join(CONDA_ENV, "Library", "bin"))
os.add_dll_directory(os.path.join(CONDA_ENV, "DLLs"))
sys.path.append(os.path.join(CONDA_ENV, "Lib", "site-packages"))
# ----------------------------------------------------------

from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.models import BEATS_LSTM

# ---------------- CONFIG (wie DBNBeatTracker) ----------------
FPS = 100

# Tighten diese Range passend zu deiner Musik -> stabiler!
MIN_BPM = 60
MAX_BPM = 190

TRANSITION_LAMBDA = 100  # wie default im DBNBeatTrackingProcessor
OBSERVATION_LAMBDA = 16  # default
THRESHOLD = 0.0          # default
CORRECT = True           # default

# Performance: Online-DBN + 1 LSTM Modell ist oft deutlich stabiler in Echtzeit
USE_SINGLE_MODEL = True  # wenn False: komplettes Online-Ensemble (mehr CPU)
# ------------------------------------------------------------

_state = {
    "init": False,
    "sr": None,

    "rnn": None,
    "dbn": None,

    # absolute sample timeline (für Mapping beat->sampleindex im Cook)
    "sample_count": 0,

    # track last reported beat to deduplicate
    "last_beat_t": -1.0,
}

def _init(sr: int):
    # ACHTUNG: RNNBeatProcessor(online=True) ist für 44100 gedacht
    _state["sr"] = sr
    _state["sample_count"] = 0

    if USE_SINGLE_MODEL:
        _state["rnn"] = RNNBeatProcessor(online=True, fps=FPS, nn_files=[BEATS_LSTM[0]])
    else:
        _state["rnn"] = RNNBeatProcessor(online=True, fps=FPS)

    # DBN im ONLINE modus: forward algorithm, stateful, frame-by-frame / block-by-block
    _state["dbn"] = DBNBeatTrackingProcessor(
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM,
        transition_lambda=TRANSITION_LAMBDA,
        observation_lambda=OBSERVATION_LAMBDA,
        threshold=THRESHOLD,
        correct=CORRECT,
        fps=FPS,
        online=True
    )

    # wichtig: online-dbns haben state -> reset am Anfang
    _state["dbn"].reset()
    _state["init"] = True

def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):

    scriptOp.isTimeSlice=False

    scriptOp.clear()

    if len(scriptOp.inputs) == 0:
        return

    in_chop = scriptOp.inputs[0]
    arr = in_chop.numpyArray()
    if arr.size == 0:
        return

    # --- Audio mono ---
    if arr.shape[0] > 1:
        audio = np.mean(arr, axis=0)
    else:
        audio = arr[0]
    audio = np.asarray(audio, dtype=np.float32)

    sr = int(getattr(in_chop, "rate", 44100))
    num_samples = int(in_chop.numSamples)

    # Output vorbereiten
    scriptOp.numSamples = num_samples
    scriptOp.appendChan("beat")
    scriptOp["beat"].vals = [0.0] * num_samples

    if not _state["init"]:
        _init(sr)

    # --- Nur NEUE Samples an die RNN füttern ---
    # Das Input-CHOP liefert seinen vollen Rolling-Buffer (z.B. 44100 Samples).
    # Bei 60fps sind aber nur ~735 Samples am Ende wirklich neu.
    # Die RNN im Online-Modus erwartet sequentielle, NICHT überlappende Audio-Blöcke.
    new_count = int(round(absTime.stepSeconds * sr))
    new_count = max(1, min(new_count, len(audio)))
    audio_new = audio[-new_count:]

    # absolute Startzeit dieses Blocks (in Sekunden)
    block_start_t = _state["sample_count"] / float(sr)
    _state["sample_count"] += new_count

    # --- Resample auf 44100 Hz fuer madmom RNN ---
    TARGET_SR = 44100
    if sr != TARGET_SR:
        n_target = int(round(len(audio_new) * TARGET_SR / sr))
        audio_new = resample(audio_new, n_target).astype(np.float32)

    # --- 1) RNN: audio_new -> activations ---
    acts = _state["rnn"](audio_new)
    acts = np.asarray(acts, dtype=np.float32)

    if acts.size == 0:
        return

    # --- 2) DBN ONLINE: activations -> beats (sekunden seit reset) ---
    # DBNBeatTrackingProcessor.process_online arbeitet stateful und inkrementell. :contentReference[oaicite:2]{index=2}
    beats_sec = _state["dbn"].process_online(acts, reset=False)
    beats_sec = np.asarray(beats_sec, dtype=np.float32)

    if beats_sec.size == 0:
        return

    # --- 3) Beats in current block mappen -> Single-Sample Impuls ---
    # Neue Samples sitzen am ENDE des Output-Buffers
    buf_offset = num_samples - new_count
    for bt in beats_sec:
        bt = float(bt)
        if bt <= _state["last_beat_t"]:
            continue
        _state["last_beat_t"] = bt

        idx = buf_offset + int(round((bt - block_start_t) * sr))
        idx = max(0, min(idx, num_samples - 1))
        scriptOp["beat"][idx] = 1.0