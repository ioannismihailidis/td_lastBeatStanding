# script_madmom_post DAT (Script CHOP)
# Author: Ioannis Mihailidis
# Email:  ioannis@studiofarbraum.com
# Web:    https://www.studiofarbraum.com
# GitHub: https://github.com/ioannismihailidis
#
# Post-processing for madmom beat detection:
# - Continues beats during silence using last known BPM
# - Half/double time multiplier
# - Beat counter with configurable beats per bar
# - Bass energy silence detection (breakdown detection)
#
# Inputs:
#   0: madmom DNN output (beat, bpm, confidence)
#   1: original audio samples (for bass energy analysis)

import numpy as np

# Bass isolation: block-average cutoff frequency in Hz
BASS_CUTOFF = 150
# Rolling audio buffer window for bass analysis (seconds)
BASS_WINDOW = 0.2

_post = {
	"last_beat_time": -1.0,
	"last_real_beat_time": -1.0,
	"synth_beat_time": -1.0,
	"last_accepted_time": -1.0,
	"running_time": 0.0,
	"beat_counter": 0,
	"reset_requested": False,
	"prev_in_beat": 0.0,
	"bass_energy": 0.0,
	"audio_buf": np.array([], dtype=np.float32),
	"audio_sr": 44100,
}

def _chan_val(chop, name, idx=0, default=0.0):
	"""Safely read a channel value from a CHOP."""
	try:
		return chop[name][idx]
	except:
		return default

def onPulse(par):
	if par.name == 'Resettact':
		_post["reset_requested"] = True

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
	scriptOp["beat_num"][0] = _post["beat_counter"]
	scriptOp["breakdown"][0] = 0.0
	scriptOp["bass_energy"][0] = _post["bass_energy"]

	if len(scriptOp.inputs) == 0:
		return

	in_chop = scriptOp.inputs[0]

	# Read input channels from madmom DNN
	in_beat = _chan_val(in_chop, "beat")
	in_bpm = _chan_val(in_chop, "bpm")
	in_confidence = _chan_val(in_chop, "confidence")

	# Parameters
	time_mult_str = scriptOp.par.Timemult.eval()
	try:
		time_mult = float(time_mult_str)
	except:
		time_mult = 1.0
	beats_per_bar = int(scriptOp.par.Beatsperbar.eval())
	max_silence = int(scriptOp.par.Maxsilence.eval())
	bass_thresh = scriptOp.par.Bassthresh.eval() if hasattr(scriptOp.par, 'Bassthresh') else 0.005
	bass_smooth = scriptOp.par.Basssmooth.eval() if hasattr(scriptOp.par, 'Basssmooth') else 0.15

	# Handle tact reset
	if _post["reset_requested"]:
		_post["beat_counter"] = 0
		_post["reset_requested"] = False

	# Track running time
	dt = absTime.stepSeconds
	_post["running_time"] += dt

	# Apply time multiplier to BPM
	output_bpm = in_bpm * time_mult

	# Compute effective interval
	effective_iv = 60.0 / output_bpm if output_bpm > 0 else 0.0

	# --- Continuously buffer audio for bass analysis (input 1) ---
	if len(scriptOp.inputs) > 1:
		audio_chop = scriptOp.inputs[1]
		arr = audio_chop.numpyArray()
		if arr.size > 0:
			# Mono downmix
			if arr.shape[0] > 1:
				mono = np.mean(arr, axis=0).astype(np.float32)
			else:
				mono = np.asarray(arr[0], dtype=np.float32)

			sr = int(getattr(audio_chop, "rate", 44100))
			_post["audio_sr"] = sr

			# Append and trim to window size
			_post["audio_buf"] = np.concatenate([_post["audio_buf"], mono])
			max_samples = int(sr * BASS_WINDOW)
			if len(_post["audio_buf"]) > max_samples:
				_post["audio_buf"] = _post["audio_buf"][-max_samples:]

	# --- Beat detection (rising edge only for real beats) ---
	beat_out = 0.0
	is_synth = False
	now = _post["running_time"]
	beat_rising = in_beat > 0.5 and _post["prev_in_beat"] <= 0.5
	_post["prev_in_beat"] = in_beat

	if beat_rising:
		# Real beat from DNN
		_post["last_real_beat_time"] = now
		_post["last_beat_time"] = now
		_post["synth_beat_time"] = now
		beat_out = 1.0
	elif effective_iv > 0 and _post["synth_beat_time"] > 0 and in_beat <= 0.5:
		# No real beat and DNN not active — check if we should generate a synthetic one
		time_since_last = now - _post["synth_beat_time"]
		if time_since_last >= effective_iv * 0.95:
			# Check max silence limit
			silence_ok = True
			if max_silence > 0 and _post["last_real_beat_time"] > 0:
				silence_duration = now - _post["last_real_beat_time"]
				silence_count = int(silence_duration / effective_iv)
				if silence_count >= max_silence:
					silence_ok = False

			if silence_ok:
				_post["synth_beat_time"] = now
				_post["last_beat_time"] = now
				beat_out = 1.0
				is_synth = True

	# --- Bass energy analysis at every beat (real and synth) ---
	if beat_out > 0:
		buf = _post["audio_buf"]
		if len(buf) > 0:
			sr = _post["audio_sr"]
			block_size = max(1, sr // BASS_CUTOFF)
			n_full = (len(buf) // block_size) * block_size

			if n_full >= block_size:
				blocks = buf[:n_full].reshape(-1, block_size)
				bass_rms = float(np.sqrt(np.mean(blocks.mean(axis=1) ** 2)))
			else:
				bass_rms = float(np.sqrt(np.mean(buf ** 2)))

			alpha = max(0.01, min(1.0, bass_smooth))
			_post["bass_energy"] = _post["bass_energy"] * (1.0 - alpha) + bass_rms * alpha

	# --- Debounce: suppress double-fires from DNN ---
	if beat_out > 0 and effective_iv > 0 and _post["last_accepted_time"] > 0:
		if (now - _post["last_accepted_time"]) < effective_iv * 0.5:
			beat_out = 0.0
			is_synth = False

	# --- Advance beat counter ---
	if beat_out > 0:
		_post["last_accepted_time"] = now
		_post["beat_counter"] = (_post["beat_counter"] % beats_per_bar) + 1

	# --- Phase calculation ---
	phase_out = 0.0
	if effective_iv > 0 and _post["last_beat_time"] > 0:
		phase_out = (now - _post["last_beat_time"]) / effective_iv % 1.0

	# --- Breakdown = bass energy below threshold (updated only at beat time) ---
	is_breakdown = _post["bass_energy"] < bass_thresh if _post["bass_energy"] > 0 else False

	# --- Update output ---
	scriptOp["beat"][0] = beat_out
	scriptOp["bpm"][0] = output_bpm
	scriptOp["beat_interval"][0] = effective_iv
	scriptOp["confidence"][0] = in_confidence
	scriptOp["phase"][0] = phase_out
	scriptOp["synth"][0] = 1.0 if is_synth else 0.0
	scriptOp["beat_num"][0] = _post["beat_counter"]
	scriptOp["breakdown"][0] = 1.0 if is_breakdown else 0.0
	scriptOp["bass_energy"][0] = _post["bass_energy"]

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
