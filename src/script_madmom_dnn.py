# script_madmom_dnn DAT (Script CHOP)
# Author: Ioannis Mihailidis
# Email:  ioannis@studiofarbraum.com
# Web:    https://www.studiofarbraum.com
# GitHub: https://github.com/ioannismihailidis
#
# Real-time beat detection using pure numpy LSTM inference (weights from madmom)
# Supports loading model weights from the tox VFS or from the filesystem.

import os, sys

# --- PATH SETUP ---
# Resolve paths relative to the tox (parent COMP), falling back to project.folder
_tox_par = me.parent().par.externaltox.eval() if me.parent() else ''
_TOX_DIR = os.path.dirname(tdu.expandPath(_tox_par)) if _tox_par else project.folder

# Add project folder so beat_detection package can be imported
if _TOX_DIR not in sys.path:
	sys.path.insert(0, _TOX_DIR)
# ------------------------------

import numpy as np
from beat_detection import BeatDetector

# Filesystem fallback path
_MODEL_DIR = os.path.join(_TOX_DIR, "beat_detection", "models")

_state = {
	"detector": None,
	"_params": None,  # track params that require re-init
}


def _load_vfs_files():
	"""Try to load model files from the parent COMP's VFS.
	Returns a dict {filename: bytes} or None if VFS is empty/unavailable.
	"""
	try:
		vfs = me.parent().vfs
		files = {}
		for item in vfs:
			files[item.name] = bytes(item.byteArray)
		if 'config.json' in files:
			return files
	except:
		pass
	return None


def _init(min_bpm, max_bpm, trans_lambda, obs_lambda, single_model, act_gate, rms_gate):
	# Try VFS first, fall back to filesystem
	vfs_files = _load_vfs_files()

	if vfs_files is not None:
		_state["detector"] = BeatDetector(
			model_files=vfs_files,
			min_bpm=min_bpm,
			max_bpm=max_bpm,
			transition_lambda=trans_lambda,
			observation_lambda=obs_lambda,
			act_gate=act_gate,
			rms_gate=rms_gate,
			single_model=single_model,
		)
	else:
		_state["detector"] = BeatDetector(
			model_dir=_MODEL_DIR,
			min_bpm=min_bpm,
			max_bpm=max_bpm,
			transition_lambda=trans_lambda,
			observation_lambda=obs_lambda,
			act_gate=act_gate,
			rms_gate=rms_gate,
			single_model=single_model,
		)
	_state["_params"] = (min_bpm, max_bpm, trans_lambda, obs_lambda, single_model)


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
		if _state["detector"] is not None:
			_state["detector"].reset()

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

	# --- Auto-reinit on parameter change ---
	current_params = (min_bpm, max_bpm, trans_lambda, obs_lambda, single_model)
	if current_params != _state["_params"] or _state["detector"] is None:
		_init(min_bpm, max_bpm, trans_lambda, obs_lambda, single_model, act_gate, rms_gate)

	detector = _state["detector"]

	# Update gates without reinit (they don't affect DBN state)
	detector.act_gate = act_gate
	detector.rms_gate = rms_gate

	# --- Get only NEW samples ---
	new_count = int(round(absTime.stepSeconds * sr))
	new_count = max(1, min(new_count, len(audio)))
	audio_new = audio[-new_count:]

	# --- Process ---
	result = detector.process(audio_new, sample_rate=sr)

	# --- Output channels (single sample) ---
	scriptOp.numSamples = 1
	scriptOp.appendChan("beat")
	scriptOp.appendChan("bpm")
	scriptOp.appendChan("beat_interval")
	scriptOp.appendChan("confidence")
	scriptOp.appendChan("phase")
	scriptOp["beat"][0] = 1.0 if result["beat"] else 0.0
	scriptOp["bpm"][0] = result["bpm"]
	scriptOp["beat_interval"][0] = result["beat_interval"]
	scriptOp["confidence"][0] = result["confidence"]
	scriptOp["phase"][0] = result["phase"]
