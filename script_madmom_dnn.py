# script_madmom_dnn DAT (Script CHOP)
# Author: Ioannis Mihailidis
# Email:  ioannis@studiofarbraum.com
# Web:    https://www.studiofarbraum.com
# GitHub: https://github.com/ioannismihailidis
#
# Real-time beat detection using madmom RNN + DBN processors

import os, sys

# --- CONDA / DLL PATH SETUP ---
CONDA_ENV = os.path.join(project.folder, "Miniconda3", "envs", "td_madmom_311")
os.add_dll_directory(os.path.join(CONDA_ENV, "Library", "bin"))
os.add_dll_directory(os.path.join(CONDA_ENV, "DLLs"))
if os.path.join(CONDA_ENV, "Lib", "site-packages") not in sys.path:
	sys.path.insert(0, os.path.join(CONDA_ENV, "Lib", "site-packages"))
# ------------------------------


import numpy as np
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.models import BEATS_LSTM

# FPS is tied to the RNN model (100 = training default)
FPS = 100
TARGET_SR = 44100
HOP_SIZE = TARGET_SR // FPS  # 441 samples per frame

# Number of beat intervals for BPM calculation
BPM_HISTORY = 8

# Minimum activation frames per DBN batch (decoupled from TD FPS)
MIN_DBN_FRAMES = 4

_state = {
	"init": False,
	"sr": None,

	"rnn": None,
	"dbn": None,

	"sample_count": 0,

	# Beat deduplication
	"last_beat_t": -1.0,
	# Last accepted beat
	"prev_beat_t": -1.0,

	# BPM from beat intervals
	"beat_intervals": [],
	"bpm": 0.0,
	"last_interval": 0.0,

	# Audio buffer for frame-aligned RNN input
	"audio_buf": np.array([], dtype=np.float32),

	# Activation buffer for DBN batching
	"act_buffer": [],

	# Auto-reinit on parameter change
	"_params": None,
}

#op("script1").isTimeSlice = False

def _init(sr, min_bpm, max_bpm, trans_lambda, obs_lambda, single_model):
	_state["sr"] = sr
	_state["sample_count"] = 0
	_state["last_beat_t"] = -1.0
	_state["prev_beat_t"] = -1.0
	_state["beat_intervals"] = []
	_state["bpm"] = 0.0
	_state["last_interval"] = 0.0
	_state["audio_buf"] = np.array([], dtype=np.float32)
	_state["act_buffer"] = []

	if single_model:
		_state["rnn"] = RNNBeatProcessor(online=True, fps=FPS, nn_files=[BEATS_LSTM[0]])
	else:
		_state["rnn"] = RNNBeatProcessor(online=True, fps=FPS)

	_state["dbn"] = DBNBeatTrackingProcessor(
		min_bpm=min_bpm,
		max_bpm=max_bpm,
		transition_lambda=trans_lambda,
		observation_lambda=obs_lambda,
		threshold=0.0,
		correct=True,
		fps=FPS,
		online=True
	)
	_state["dbn"].reset()
	_state["init"] = True

def _update_bpm(interval, min_bpm, max_bpm):
	"""Simple BPM calculation from recent beat intervals."""
	inst_bpm = 60.0 / interval
	if inst_bpm < min_bpm or inst_bpm > max_bpm:
		return

	_state["beat_intervals"].append(interval)
	if len(_state["beat_intervals"]) > BPM_HISTORY:
		_state["beat_intervals"] = _state["beat_intervals"][-BPM_HISTORY:]

	# Median of recent intervals -> BPM
	intervals = sorted(_state["beat_intervals"])
	median_iv = intervals[len(intervals) // 2]
	_state["bpm"] = round(60.0 / median_iv, 2)

def _addDnnParams(page):
	"""Create beat detection parameters on the given page."""
	p = page.appendFloat('Actgate', label='Activation Gate')
	p[0].default = 0.15
	p[0].val = 0.15
	p[0].min = 0.0
	p[0].max = 1.0
	p[0].normMin = 0.0
	p[0].normMax = 0.5
	p[0].clampMin = True
	p[0].clampMax = True

	p = page.appendFloat('Rmsgate', label='RMS Gate')
	p[0].default = 0.005
	p[0].val = 0.005
	p[0].min = 0.0
	p[0].max = 0.1
	p[0].normMin = 0.0
	p[0].normMax = 0.05
	p[0].clampMin = True
	p[0].clampMax = True

	p = page.appendInt('Minbpm', label='Min BPM')
	p[0].default = 60
	p[0].val = 60
	p[0].min = 30
	p[0].max = 200
	p[0].normMin = 30
	p[0].normMax = 200
	p[0].clampMin = True
	p[0].clampMax = True

	p = page.appendInt('Maxbpm', label='Max BPM')
	p[0].default = 190
	p[0].val = 190
	p[0].min = 60
	p[0].max = 300
	p[0].normMin = 60
	p[0].normMax = 300
	p[0].clampMin = True
	p[0].clampMax = True

	p = page.appendInt('Translambda', label='Transition Lambda')
	p[0].default = 100
	p[0].val = 100
	p[0].min = 1
	p[0].max = 300
	p[0].normMin = 1
	p[0].normMax = 300
	p[0].clampMin = True
	p[0].clampMax = True

	p = page.appendInt('Obslambda', label='Observation Lambda')
	p[0].default = 16
	p[0].val = 16
	p[0].min = 1
	p[0].max = 64
	p[0].normMin = 1
	p[0].normMax = 64
	p[0].clampMin = True
	p[0].clampMax = True

	p = page.appendToggle('Singlemodel', label='Single LSTM Model')
	p[0].default = True
	p[0].val = True

	p = page.appendPulse('Reset', label='Reset')

def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('Beat Detection')
	_addDnnParams(page)

	# --- Promote parameters to parent COMP ---
	parent_comp = scriptOp.parent()
	for pg in parent_comp.customPages:
		if pg.name == 'Beat Detection':
			pg.destroy()
	parent_page = parent_comp.appendCustomPage('Beat Detection')
	_addDnnParams(parent_page)

	# Bind scriptOp value parameters to parent
	for name in ['Actgate', 'Rmsgate', 'Minbpm', 'Maxbpm', 'Translambda', 'Obslambda', 'Singlemodel']:
		getattr(scriptOp.par, name).bindExpr = "parent().par." + name

def onPulse(par):
	if par.name == 'Reset':
		_state["sample_count"] = 0
		_state["last_beat_t"] = -1.0
		_state["prev_beat_t"] = -1.0
		_state["beat_intervals"] = []
		_state["bpm"] = 0.0
		_state["last_interval"] = 0.0
		_state["audio_buf"] = np.array([], dtype=np.float32)
		_state["act_buffer"] = []
		if _state["dbn"] is not None:
			_state["dbn"].reset()

def onCook(scriptOp):
	scriptOp.isTimeSlice = False
	scriptOp.clear()

	if len(scriptOp.inputs) == 0:
		return

	in_chop = scriptOp.inputs[0]
	arr = in_chop.numpyArray()
	if arr.size == 0:
		return

	# --- Mono downmix ---
	if arr.shape[0] > 1:
		audio = np.mean(arr, axis=0)
	else:
		audio = arr[0]
	audio = np.asarray(audio, dtype=np.float32)

	sr = int(getattr(in_chop, "rate", 44100))

	# --- Parameters ---
	act_gate = scriptOp.par.Actgate.eval()
	rms_gate = scriptOp.par.Rmsgate.eval()
	min_bpm = int(scriptOp.par.Minbpm.eval())
	max_bpm = int(scriptOp.par.Maxbpm.eval())
	trans_lambda = int(scriptOp.par.Translambda.eval())
	obs_lambda = int(scriptOp.par.Obslambda.eval())
	single_model = bool(scriptOp.par.Singlemodel.eval())

	# --- Output channels (single sample) ---
	scriptOp.numSamples = 1
	scriptOp.appendChan("beat")
	scriptOp.appendChan("bpm")
	scriptOp.appendChan("beat_interval")
	scriptOp.appendChan("confidence")
	scriptOp.appendChan("phase")
	scriptOp["beat"][0] = 0.0
	scriptOp["bpm"][0] = _state["bpm"]
	scriptOp["beat_interval"][0] = _state["last_interval"]
	scriptOp["confidence"][0] = 0.0
	scriptOp["phase"][0] = 0.0

	# --- Auto-reinit on parameter change ---
	current_params = (min_bpm, max_bpm, trans_lambda, obs_lambda, single_model)
	if current_params != _state["_params"]:
		_state["_params"] = current_params
		_state["init"] = False

	if not _state["init"]:
		_init(sr, min_bpm, max_bpm, trans_lambda, obs_lambda, single_model)

	# --- Get only NEW samples ---
	new_count = int(round(absTime.stepSeconds * sr))
	new_count = max(1, min(new_count, len(audio)))
	audio_new = audio[-new_count:]

	_state["sample_count"] += new_count

	# --- Resample to 44100 Hz (linear, no scipy) ---
	if sr != TARGET_SR:
		n_target = int(round(len(audio_new) * TARGET_SR / sr))
		if n_target > 0 and len(audio_new) > 1:
			x_old = np.linspace(0, 1, len(audio_new))
			x_new = np.linspace(0, 1, n_target)
			audio_new = np.interp(x_new, x_old, audio_new).astype(np.float32)

	# --- Frame-aligned audio buffer ---
	# Accumulate audio and only feed exact multiples of HOP_SIZE to the RNN.
	# This ensures the RNN always produces an integer number of frames,
	# regardless of TouchDesigner's cook rate.
	_state["audio_buf"] = np.concatenate([_state["audio_buf"], audio_new])

	n_frames = len(_state["audio_buf"]) // HOP_SIZE
	if n_frames == 0:
		if _state["bpm"] > 0 and _state["prev_beat_t"] > 0:
			current_t = _state["sample_count"] / float(sr)
			expected_iv = 60.0 / _state["bpm"]
			phase = (current_t - _state["prev_beat_t"]) / expected_iv % 1.0
			scriptOp["phase"][0] = phase
		return

	n_samples_feed = n_frames * HOP_SIZE
	rnn_input = _state["audio_buf"][:n_samples_feed]
	_state["audio_buf"] = _state["audio_buf"][n_samples_feed:]

	# --- 1) RNN: frame-aligned audio -> activations ---
	acts = _state["rnn"](rnn_input)
	acts = np.atleast_1d(np.asarray(acts, dtype=np.float32))

	if acts.size == 0:
		if _state["bpm"] > 0 and _state["prev_beat_t"] > 0:
			current_t = _state["sample_count"] / float(sr)
			expected_iv = 60.0 / _state["bpm"]
			phase = (current_t - _state["prev_beat_t"]) / expected_iv % 1.0
			scriptOp["phase"][0] = phase
		return

	# --- Confidence + RMS ---
	peak_act = float(acts.max())
	scriptOp["confidence"][0] = peak_act
	rms = float(np.sqrt(np.mean(rnn_input ** 2)))
	gated = rms < rms_gate or peak_act < act_gate

	# --- 2) Always buffer activations (DBN needs continuous stream) ---
	_state["act_buffer"].append(acts)

	# Only send to DBN when enough frames have accumulated
	total_frames = sum(a.size for a in _state["act_buffer"])
	if total_frames < MIN_DBN_FRAMES:
		if _state["bpm"] > 0 and _state["prev_beat_t"] > 0:
			current_t = _state["sample_count"] / float(sr)
			expected_iv = 60.0 / _state["bpm"]
			phase = (current_t - _state["prev_beat_t"]) / expected_iv % 1.0
			scriptOp["phase"][0] = phase
		return

	# --- 3) DBN: buffered activations -> beat times ---
	batched_acts = np.concatenate(_state["act_buffer"], axis=0)
	_state["act_buffer"] = []

	beats_sec = _state["dbn"].process_online(batched_acts, reset=False)
	beats_sec = np.asarray(beats_sec, dtype=np.float32)

	if beats_sec.size == 0 or gated:
		if _state["bpm"] > 0 and _state["prev_beat_t"] > 0:
			current_t = _state["sample_count"] / float(sr)
			expected_iv = 60.0 / _state["bpm"]
			phase = (current_t - _state["prev_beat_t"]) / expected_iv % 1.0
			scriptOp["phase"][0] = phase
		return

	# --- 4) Beat processing ---
	beat_detected = False
	for bt in beats_sec:
		bt = float(bt)
		if bt <= _state["last_beat_t"]:
			continue
		_state["last_beat_t"] = bt

		# Compute BPM from interval
		if _state["prev_beat_t"] > 0:
			interval = bt - _state["prev_beat_t"]
			if interval > 0:
				_state["last_interval"] = round(interval, 4)
				_update_bpm(interval, min_bpm, max_bpm)

		_state["prev_beat_t"] = bt
		beat_detected = True

	# --- Update output ---
	if beat_detected:
		scriptOp["beat"][0] = 1.0
	scriptOp["bpm"][0] = _state["bpm"]
	scriptOp["beat_interval"][0] = _state["last_interval"]

	if _state["bpm"] > 0 and _state["prev_beat_t"] > 0:
		current_t = _state["sample_count"] / float(sr)
		expected_iv = 60.0 / _state["bpm"]
		phase = (current_t - _state["prev_beat_t"]) / expected_iv % 1.0
		scriptOp["phase"][0] = phase
