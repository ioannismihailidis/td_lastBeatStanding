# last beat standing - a realtime beat detector

Real-time beat tracking component for TouchDesigner using LSTM neural networks and a Dynamic Bayesian Network beat tracker. Pure numpy — no external dependencies required.

## Overview

**lastBeatStanding** is a self-contained TouchDesigner component (`.tox`) that provides robust, real-time beat detection from a live audio stream. The neural network weights are embedded in the tox's Virtual File System, so it works out of the box with stock TouchDesigner — no conda, no pip, no external packages.

## Requirements

- TouchDesigner 2025.32280+
- tested with Windows 11 or MacOS 15.5

## Quick Start

1. Download the latest `td_lastBeatStanding.tox` from the [Releases](https://github.com/ioannismihailidis/td_lastBeatStanding/releases) page.
2. Drag `td_lastBeatStanding.tox` into your TouchDesigner project.
3. Connect an audio CHOP (e.g. Audio Device In, Audio File In) to the component's input.
4. The component outputs a single CHOP with `beat`, `bpm`, `phase`, `breakdown`, and other channels.

That's it — no installation, no environment setup.

To try the included example, clone the repository and open `td_lastBeatStanding_example.toe`.

<img width="1891" height="837" alt="image" src="https://github.com/user-attachments/assets/5824eec5-4d71-45ab-8710-2b183fa621e2" />

## How It Works

The beat detector runs a three-stage pipeline entirely in Python/numpy:

1. **Preprocessing** — Audio is framed, windowed, FFT'd, and passed through a logarithmic filterbank to produce spectral features at 100 FPS.
2. **LSTM Inference** — Features are fed through pre-trained LSTM neural networks (exported from [madmom](https://github.com/CPJKU/madmom)) to produce beat activation values.
3. **DBN Beat Tracking** — A Dynamic Bayesian Network tracks tempo and phase over time, producing stable beat positions even through noisy activations.

Post-processing adds synthetic beat continuation, time multiplier, bar counting, and breakdown detection.

## Architecture

```
Audio In --> [Preprocessing] --> [LSTM Ensemble] --> [DBN Tracker] --> [Post Processing] --> Output
                                                                             ^
                                                             Audio In -------+
                                                          (bass energy analysis)
```

### Model files

The LSTM weights are stored as `.npz` files (numpy archives). They can be loaded from:

- **TouchDesigner VFS** (preferred) — embedded in the `.tox`, fully self-contained
- **Filesystem** — from the `beat_detection/models/` directory as a fallback

The TD script auto-detects VFS first, then falls back to the filesystem.

## Standalone Python Usage

The beat detection library works independently of TouchDesigner:

```python
from beat_detection import BeatDetector, PostProcessor

detector = BeatDetector(model_dir='beat_detection/models/', single_model=True)
post = PostProcessor()

# Feed audio chunks (e.g. from a microphone or file)
result = detector.process(audio_chunk, sample_rate=44100)
result = post.process(result, dt=1/60, audio=audio_chunk)

if result['beat']:
    print(f"Beat! BPM: {result['bpm']}")
```

**Dependencies**: numpy only.

Run the included example:

```bash
pip install numpy soundfile
python -m beat_detection.example Assets/cyba_-_yellow.mp3
```

## Re-exporting Models

The `.npz` weight files are already included. If you need to re-export from madmom (e.g. for different models), run the export script in an environment with madmom and onnx installed:

```bash
pip install madmom onnx numpy
python -m beat_detection.export_models
```

This only needs to be done once. The exported weights are then used at runtime without madmom.

## Component Parameters

All parameters are promoted to the parent component. Adjusting them on the COMP controls the internal Script CHOPs via parameter binding.

### Beat Detection

| Parameter            | Type    | Default | Range     | Description                                                                 |
|----------------------|---------|---------|-----------|-----------------------------------------------------------------------------|
| Activation Gate      | Float   | 0.15    | 0.0 - 1.0 | Minimum RNN activation to accept a beat. Higher = less sensitive.           |
| RMS Gate             | Float   | 0.005   | 0.0 - 0.1 | Minimum audio RMS level. Suppresses beats during silence.                   |
| Min BPM              | Int     | 60      | 30 - 200  | Lower BPM limit for the DBN beat tracker.                                   |
| Max BPM              | Int     | 190     | 60 - 300  | Upper BPM limit for the DBN beat tracker.                                   |
| Transition Lambda    | Int     | 100     | 1 - 300   | DBN tempo transition smoothness. Higher = more stable tempo.                |
| Observation Lambda   | Int     | 16      | 1 - 64    | DBN observation weight. Higher = stronger trust in RNN activations.         |
| Single LSTM Model    | Toggle  | On      | --        | Use only the first LSTM model (faster) instead of the full 8-model ensemble.|
| Reset                | Pulse   | --      | --        | Reset all beat detection state and the DBN processor.                       |

### Post Processing

| Parameter        | Type   | Default | Range      | Description                                                                  |
|------------------|--------|---------|------------|------------------------------------------------------------------------------|
| Time Multiplier  | Menu   | 1x     | 0.5x / 1x / 2x / 4x | Scales the detected BPM. 0.5x = half time, 2x = double time.       |
| Beats Per Bar    | Int    | 4       | 1 - 16    | Number of beats per bar for the beat counter.                                |
| Reset Tact       | Pulse  | --      | --         | Reset the beat counter to 0.                                                 |
| Max Synth Beats  | Int    | 16      | 0 - 128   | Maximum synthetic beats before stopping. 0 = unlimited.                      |
| Bass Threshold   | Float  | 0.005   | 0.0 - 0.1 | Bass energy level below which a breakdown is detected.                       |
| Bass Smoothing   | Float  | 0.15    | 0.01 - 1.0 | Exponential smoothing factor for bass energy. Lower = slower response.      |

### Output Channels

| Channel         | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `beat`          | 1.0 on any beat (real or synthetic), 0.0 otherwise              |
| `bpm`           | BPM after time multiplier is applied                             |
| `beat_interval` | Effective beat interval in seconds (after time multiplier)       |
| `confidence`    | Peak LSTM activation value (0.0 - 1.0)                          |
| `phase`         | Position within current beat cycle (0.0 - 1.0)                  |
| `synth`         | 1.0 when the beat is synthetic, 0.0 for real beats              |
| `beat_num`      | Current beat position within the bar (1 to Beats Per Bar)        |
| `breakdown`     | 1.0 when bass energy is below threshold (breakdown detected)    |
| `bass_energy`   | Smoothed bass energy level                                       |

## Demo

Demo track: [cyba - Nostalgia](https://ccmixter.org/files/cyba/60166) from ccMixter.

## License

This project's source code is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

The pre-trained model weights (`beat_detection/models/*.npz`) are derived from [madmom](https://github.com/CPJKU/madmom) and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial use of the model weights requires permission from the original authors — see [LICENSE](LICENSE) for details.
