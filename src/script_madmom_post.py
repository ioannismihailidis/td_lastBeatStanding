# script_madmom_post DAT (Script CHOP)
# Author: Ioannis Mihailidis
# Email:  ioannis@studiofarbraum.com
# Web:    https://www.studiofarbraum.com
# GitHub: https://github.com/ioannismihailidis
#
# Post-processing for beat detection:
# - Continues beats during silence using last known BPM
# - Half/double time multiplier
# - Beat counter with configurable beats per bar
# - Bass energy silence detection (breakdown detection)
#
# Inputs:
#   0: DNN output (beat, bpm, confidence)
#   1: original audio samples (for bass energy analysis)

import os, sys

import numpy as np
from post_processor import PostProcessor

_post_proc = PostProcessor()


def _chan_val(chop, name, idx=0, default=0.0):
	"""Safely read a channel value from a CHOP."""
	try:
		return chop[name][idx]
	except:
		return default

def onPulse(par):
	if par.name == 'Resettact':
		_post_proc.reset_beat_counter()

def onCook(scriptOp):
	scriptOp.isTimeSlice = False
	scriptOp.clear()

	# --- Output channels (single sample) ---
	scriptOp.numSamples = 1
	scriptOp.appendChan("beat")
	scriptOp.appendChan("bpm")
	scriptOp.appendChan("beat_interval")
	scriptOp.appendChan("confidence")
	scriptOp.appendChan("phase")
	scriptOp.appendChan("synth")
	scriptOp.appendChan("beat_num")
	scriptOp.appendChan("breakdown")
	scriptOp.appendChan("bass_energy")
	scriptOp["beat"][0] = 0.0
	scriptOp["bpm"][0] = 0.0
	scriptOp["beat_interval"][0] = 0.0
	scriptOp["confidence"][0] = 0.0
	scriptOp["phase"][0] = 0.0
	scriptOp["synth"][0] = 0.0
	scriptOp["beat_num"][0] = _post_proc.beat_counter
	scriptOp["breakdown"][0] = 0.0
	scriptOp["bass_energy"][0] = _post_proc.bass_energy

	if len(scriptOp.inputs) == 0:
		return

	in_chop = scriptOp.inputs[0]

	# Read input channels from DNN
	in_beat = _chan_val(in_chop, "beat")
	in_bpm = _chan_val(in_chop, "bpm")
	in_confidence = _chan_val(in_chop, "confidence")

	# Parameters — update PostProcessor settings each cook
	time_mult_str = scriptOp.par.Timemult.eval()
	try:
		_post_proc.time_mult = float(time_mult_str)
	except:
		_post_proc.time_mult = 1.0
	_post_proc.beats_per_bar = int(scriptOp.par.Beatsperbar.eval())
	_post_proc.max_synth_beats = int(scriptOp.par.Maxsilence.eval())
	_post_proc.bass_thresh = scriptOp.par.Bassthresh.eval() if hasattr(scriptOp.par, 'Bassthresh') else 0.005
	_post_proc.bass_smooth = scriptOp.par.Basssmooth.eval() if hasattr(scriptOp.par, 'Basssmooth') else 0.15

	# Build beat_result dict matching BeatDetector output
	beat_result = {
		'beat': in_beat > 0.5,
		'bpm': float(in_bpm),
		'confidence': float(in_confidence),
	}

	# Get audio for bass analysis (input 1)
	audio = None
	audio_sr = 44100
	if len(scriptOp.inputs) > 1:
		audio_chop = scriptOp.inputs[1]
		arr = audio_chop.numpyArray()
		if arr.size > 0:
			if arr.shape[0] > 1:
				audio = np.mean(arr, axis=0).astype(np.float32)
			else:
				audio = np.asarray(arr[0], dtype=np.float32)
			audio_sr = int(getattr(audio_chop, "rate", 44100))

	# Process
	dt = absTime.stepSeconds
	result = _post_proc.process(
		beat_result=beat_result,
		dt=dt,
		audio=audio,
		sample_rate=audio_sr,
	)

	# --- Update output ---
	scriptOp["beat"][0] = 1.0 if result["beat"] else 0.0
	scriptOp["bpm"][0] = result["bpm"]
	scriptOp["beat_interval"][0] = result["beat_interval"]
	scriptOp["confidence"][0] = result["confidence"]
	scriptOp["phase"][0] = result["phase"]
	scriptOp["synth"][0] = 1.0 if result["synth"] else 0.0
	scriptOp["beat_num"][0] = result["beat_num"]
	scriptOp["breakdown"][0] = 1.0 if result["breakdown"] else 0.0
	scriptOp["bass_energy"][0] = result["bass_energy"]

def _addPostParams(page):
	"""Create post-processing parameters on the given page."""
	# --- Beat ---
	p = page.appendMenu('Timemult', label='Time Multiplier')
	p[0].menuNames = ['0.5', '1.0', '2.0', '4.0']
	p[0].menuLabels = ['0.5x', '1x', '2x', '4x']
	p[0].default = '1.0'
	p[0].val = '1.0'

	p = page.appendInt('Beatsperbar', label='Beats Per Bar')
	p[0].default = 4
	p[0].val = 4
	p[0].min = 1
	p[0].max = 16
	p[0].clampMin = True
	p[0].clampMax = True

	p = page.appendPulse('Resettact', label='Reset Tact')

	p = page.appendInt('Maxsilence', label='Max Synth Beats')
	p[0].default = 16
	p[0].val = 16
	p[0].min = 0
	p[0].max = 128
	p[0].clampMin = True
	p[0].clampMax = True

	# --- Breakdown Detection ---
	p = page.appendFloat('Bassthresh', label='Bass Threshold')
	p[0].startSection = True
	p[0].default = 0.005
	p[0].val = 0.005
	p[0].min = 0.0
	p[0].max = 0.1
	p[0].normMax = 0.05
	p[0].clampMin = True

	p = page.appendFloat('Basssmooth', label='Bass Smoothing')
	p[0].default = 0.15
	p[0].val = 0.15
	p[0].min = 0.01
	p[0].max = 1.0
	p[0].normMax = 0.5
	p[0].clampMin = True
	p[0].clampMax = True

def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('Post Processing')
	_addPostParams(page)

	# --- Promote parameters to parent COMP ---
	parent_comp = scriptOp.parent()
	for pg in parent_comp.customPages:
		if pg.name == 'Post Processing':
			pg.destroy()
	parent_page = parent_comp.appendCustomPage('Post Processing')
	_addPostParams(parent_page)

	# Bind scriptOp value parameters to parent
	for name in ['Timemult', 'Beatsperbar', 'Maxsilence', 'Bassthresh', 'Basssmooth']:
		getattr(scriptOp.par, name).bindExpr = "parent().par." + name
