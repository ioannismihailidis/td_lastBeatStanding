# td_lastBeatStanding

Real-time beat tracking component for TouchDesigner using madmom's RNN + DBN neural network processors. Distributed as a ready-to-use `.tox` component.

**Author:** Ioannis Mihailidis
**Email:** ioannis@studiofarbraum.com
**Web:** https://www.studiofarbraum.com
**GitHub:** https://github.com/ioannismihailidis

## Overview

**lastBeatStanding** is a self-contained TouchDesigner component (`.tox`) that provides robust, real-time beat detection from a live audio stream. Drop the `.tox` into your project, connect an audio source, and all parameters are exposed on the component's custom parameter pages.

Inside the component, two Script CHOPs work together:

1. **script_madmom_dnn** -- Core beat detection using madmom's recurrent neural network (RNN) and Dynamic Bayesian Network (DBN) processors.
2. **script_madmom_post** -- Post-processing that adds synthetic beat continuation, time multiplier, beat counting, and bass energy-based breakdown detection.

## Installation

1. Copy `td_lastBeatStanding.tox`, `script_madmom_dnn.py`, and `script_madmom_post.py` into your project folder.
2. Drag and drop the `.tox` into your TouchDesigner project.
3. Miniconda and all Python dependencies (`madmom`, `numpy`) are automatically installed into the project folder via [tdPyEnvManager](https://derivative.ca/community-post/introducing-touchdesigner-python-environment-manager-tdpyenvmanager/72024).
4. Connect an audio CHOP to the component's input.
5. All parameters are available on the component's custom parameter pages (**Beat Detection** and **Post Processing**).

## Usage

Connect an audio CHOP (e.g. Audio Device In, Audio File In) to the component's input. The component outputs a single CHOP with beat, bpm, phase, breakdown, and other channels that you can use to drive visuals, lighting, or any other downstream logic.

For breakdown detection, the original audio is also routed internally to the post-processing stage for bass energy analysis.

## Signal Flow (internal)

```
Audio In --> [script_madmom_dnn] --> [script_madmom_post] --> Output
                                          ^
                          Audio In --------+
                       (input 1: bass analysis)
```

## Component Parameters

All parameters are promoted to the parent component. Adjusting them on the COMP controls the internal Script CHOPs via parameter binding.

### script_madmom_dnn

Performs real-time beat detection by feeding audio through madmom's `RNNBeatProcessor` (online mode) and `DBNBeatTrackingProcessor` (online mode). Audio is frame-aligned to the RNN's expected hop size (441 samples at 100 FPS). Activations are batched before being sent to the DBN. BPM is calculated from the median of recent beat intervals.

The processor auto-reinitializes when detection parameters change.

#### Output Channels

| Channel         | Description                                      |
|-----------------|--------------------------------------------------|
| `beat`          | 1.0 on beat frame, 0.0 otherwise                |
| `bpm`           | Current BPM (median of recent intervals)         |
| `beat_interval` | Time in seconds between last two beats           |
| `confidence`    | Peak RNN activation value (0.0 - 1.0)           |
| `phase`         | Position within current beat cycle (0.0 - 1.0)  |

#### Parameters (Beat Detection)

| Parameter            | Type    | Default | Range     | Description                                                                 |
|----------------------|---------|---------|-----------|-----------------------------------------------------------------------------|
| Activation Gate      | Float   | 0.15    | 0.0 - 1.0 | Minimum RNN activation to accept a beat. Higher = less sensitive.           |
| RMS Gate             | Float   | 0.005   | 0.0 - 0.1 | Minimum audio RMS level. Suppresses beats during silence.                   |
| Min BPM              | Int     | 60      | 30 - 200  | Lower BPM limit for the DBN beat tracker.                                   |
| Max BPM              | Int     | 190     | 60 - 300  | Upper BPM limit for the DBN beat tracker.                                   |
| Transition Lambda    | Int     | 100     | 1 - 300   | DBN tempo transition smoothness. Higher = more stable tempo.                |
| Observation Lambda   | Int     | 16      | 1 - 64    | DBN observation weight. Higher = stronger trust in RNN activations.         |
| Single LSTM Model    | Toggle  | On      | --        | Use only the first LSTM model (faster) instead of the full ensemble.        |
| Reset                | Pulse   | --      | --        | Reset all beat detection state and the DBN processor.                       |

---

### script_madmom_post

Post-processes the DNN beat output to provide synthetic beat continuation during silence, half/double time multiplier, a beat counter (bar position), and bass energy-based breakdown detection.

Synthetic beats are generated using the last known BPM when the DNN stops detecting beats (e.g. during a breakdown or quiet section). A debounce guard prevents double-triggering from the DNN.

Bass energy is measured at each beat (real and synthetic) from a rolling audio buffer using a block-averaging low-pass filter (~150 Hz cutoff) with exponential smoothing.

#### Inputs

| Input | Description                                          |
|-------|------------------------------------------------------|
| 0     | Output of script_madmom_dnn (beat, bpm, confidence)  |
| 1     | Original audio samples (for bass energy analysis)    |

#### Output Channels

| Channel         | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `beat`          | 1.0 on any beat (real or synthetic), 0.0 otherwise              |
| `bpm`           | BPM after time multiplier is applied                             |
| `beat_interval` | Effective beat interval in seconds (after time multiplier)       |
| `confidence`    | Passthrough of DNN confidence                                    |
| `phase`         | Position within current beat cycle (0.0 - 1.0)                  |
| `synth`         | 1.0 only when the beat is synthetic, 0.0 for real DNN beats     |
| `beat_num`      | Current beat position within the bar (1 to Beats Per Bar)        |
| `breakdown`     | 1.0 when bass energy is below threshold (breakdown detected)    |
| `bass_energy`   | Smoothed bass energy level                                       |

#### Parameters (Post Processing)

| Parameter        | Type   | Default | Range      | Description                                                                  |
|------------------|--------|---------|------------|------------------------------------------------------------------------------|
| Time Multiplier  | Menu   | 1x     | 0.5x / 1x / 2x / 4x | Scales the detected BPM. 0.5x = half time, 2x = double time.       |
| Beats Per Bar    | Int    | 4       | 1 - 16    | Number of beats per bar for the beat counter.                                |
| Reset Tact       | Pulse  | --      | --         | Reset the beat counter to 0.                                                 |
| Max Synth Beats  | Int    | 16      | 0 - 128   | Maximum synthetic beats to generate before stopping. 0 = unlimited.          |
| Bass Threshold   | Float  | 0.005   | 0.0 - 0.1 | Bass energy level below which a breakdown is detected.                       |
| Bass Smoothing   | Float  | 0.15    | 0.01 - 1.0 | Exponential smoothing factor for bass energy. Lower = slower response.      |


## Demo

Demo track: [cyba - Nostalgia](https://ccmixter.org/files/cyba/60166) from ccMixter.

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html). See [LICENSE](LICENSE) for details.
