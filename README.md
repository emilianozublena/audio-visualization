# Timbral Space Visualizer

A Python script that analyzes audio and renders a **3D animated visualization** of its timbral characteristics. The result is a point cloud tracing a path through "timbral space" — a 3D coordinate system where each axis represents a different acoustic feature extracted from the audio signal.

---

## What is "Timbral Space"?

Every sound has a **timbre** — the quality that lets you tell apart a flute from a guitar even when they play the same note at the same volume. Timbre is shaped by the distribution of energy across frequencies and how it changes over time.

This script takes three measurable aspects of timbre and maps them to the three axes of a 3D space:

| Axis | Feature | What it measures |
|------|---------|-----------------|
| **X** | Spectral Centroid (kHz) | The "center of mass" of the frequency spectrum. Higher values mean the sound is **brighter** (more high-frequency energy); lower values mean it is **darker**. A cymbal crash has a high centroid; a bass drum has a low one. |
| **Y** | Spectral Bandwidth (kHz) | How **spread out** the energy is around the centroid. A pure sine wave has near-zero bandwidth (all energy at one frequency). A noisy sound like white noise has very high bandwidth. This is also called **spectral spread**. |
| **Z** | RMS Loudness (dBFS) | The overall **energy level** of the sound at each moment, measured in decibels relative to full scale. Louder moments sit higher on this axis; silence drops to the bottom. This correlates with perceived **tonality/intensity**. |
| **Color** | All three features | Each feature drives one HSV color channel: centroid → **hue** (red=dark, blue=bright), bandwidth → **saturation** (focused=muted, spread=vivid), loudness → **value** (quiet=dim, loud=bright). So color reinforces the same information as position. |

As the audio plays, each analysis frame becomes a **dot** positioned in this 3D space. Consecutive dots are connected by a **line**, forming a trajectory — the path the sound traces through timbral space over time.

---

## How the Audio Analysis Works

The script uses [librosa](https://librosa.org/), a widely-used Python library for audio and music analysis.

### 1. Loading the audio

```python
y, sr = librosa.load(audio_path, sr=None, mono=True)
```

- `y` is a 1D numpy array of audio samples (converted to mono if stereo).
- `sr` is the sample rate (e.g., 44100 Hz). Using `sr=None` preserves the file's native sample rate rather than resampling.

### 2. Frame-by-frame feature extraction

The audio is divided into overlapping frames using a sliding window. Two parameters control this:

- **`N_FFT = 2048`** — The size of each FFT window in samples. At 44100 Hz, this is about 46 ms of audio. Larger windows give better frequency resolution but worse time resolution.
- **`HOP_LENGTH = 512`** — How far the window advances between frames, in samples. At 44100 Hz, this means one analysis frame every ~11.6 ms, giving roughly 86 frames per second of audio.

For each frame, three features are computed:

#### Spectral Centroid

```python
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
```

The spectral centroid is the weighted mean of frequencies present in the signal, where the weights are the magnitudes of each frequency bin. It is calculated as:

```
centroid = sum(frequency[k] * magnitude[k]) / sum(magnitude[k])
```

The result is in Hz and is converted to kHz for display. Think of it as "where is the average frequency energy located?" — a single number that summarizes the brightness of the sound at that moment.

#### Spectral Bandwidth

```python
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=512)
```

The spectral bandwidth (or spectral spread) measures how wide the frequency distribution is around the centroid. It is the weighted standard deviation of frequencies:

```
bandwidth = sqrt( sum(magnitude[k] * (frequency[k] - centroid)^2) / sum(magnitude[k]) )
```

A tonal sound with energy concentrated at a single frequency has low bandwidth. A noisy, broadband sound has high bandwidth.

#### RMS Energy (Loudness)

```python
rms = librosa.feature.rms(y=y, hop_length=512)
rms_db = librosa.amplitude_to_db(rms, ref=np.max)
```

RMS (Root Mean Square) energy is the standard measure of signal amplitude:

```
rms = sqrt( (1/N) * sum(sample[k]^2) )
```

It is then converted to decibels relative to the maximum RMS value in the file (`ref=np.max`), so the loudest moment is at 0 dBFS and quieter moments are negative.

### 3. Time alignment

```python
times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)
```

Each analysis frame is mapped to a timestamp (in seconds) so the visualization can be synchronized with audio playback.

---

## How the Visualization Works

The visualization is built with [matplotlib](https://matplotlib.org/) using its 3D projection capabilities and `FuncAnimation` for frame-by-frame animation.

### Visual elements

Each animation frame renders three things:

1. **Trail line** — A thin white line connecting the last `TRAIL_LENGTH` (70) analysis frames. This shows the recent trajectory through timbral space.

2. **Trail dots** — Colored scatter points along the trail. Their size increases from oldest (small, 6px) to newest (large, 38px), creating a "comet tail" effect. Color is derived from all three features via HSV mapping (see table above).

3. **Current dot** — A larger white dot (110px) with a subtle border marking the current position in timbral space.

### Camera rotation

The 3D camera slowly rotates around the scene at `ROTATE_SPEED` (0.35) degrees per animation frame, giving a sense of depth as the trajectory builds up.

### Dark theme

The entire plot uses a dark color scheme (`#080810` background) with subtle grid lines, designed so the colored data points stand out.

---

## Three Scripts

### 1. `visualize.py` — Single file visualization

```bash
python visualize.py birdsong.wav                    # live window + audio
python visualize.py birdsong.wav --export out.mp4   # save MP4 with audio
```

- **Live mode** (default): Opens an interactive matplotlib window and plays the audio through your speakers using [sounddevice](https://python-sounddevice.readthedocs.io/). The animation syncs to **wall-clock time**, so even if frames drop, the dot stays in sync with what you hear. You can rotate/zoom the 3D plot with your mouse.
- **Export mode** (`--export`): Renders frames deterministically, pipes raw video to `ffmpeg`, then muxes the original audio on top (AAC at 192 kbps). No window opens.

### 2. `compare.py` — Side-by-side comparison of two files

```bash
python compare.py bird1.wav bird2.wav                    # live window
python compare.py bird1.wav bird2.wav --export comp.mp4  # save MP4
```

- Both tracks animate **simultaneously in the same 3D space** with shared axis scales so their trajectories are directly comparable.
- Each track gets its own **color palette** — Track A in cyan/blue, Track B in orange/red — so you can tell them apart at a glance.
- In export mode, both audio tracks are mixed together into the MP4's audio stream.

### 3. `live.py` — Real-time microphone visualization

```bash
python live.py                    # listen indefinitely
python live.py --duration 30      # stop after 30 seconds
python live.py --window 15        # 15-second rolling window
```

- Captures audio from the **default input device** (microphone) and animates the timbral space in real time.
- Uses a rolling window (default 10 seconds) — old data scrolls off as new data arrives.
- Axis limits **auto-scale** smoothly using an exponential moving average, so the plot adapts to whatever you're recording without jumping around.
- A pulsing red "LIVE" indicator confirms the system is listening.

---

## Configuration

All tuneable parameters are defined as constants at the top of the script:

| Constant | Default | Description |
|----------|---------|-------------|
| `HOP_LENGTH` | `512` | Analysis window hop in samples. Smaller = more frames = smoother animation but slower export. |
| `N_FFT` | `2048` | FFT window size. Larger = better frequency resolution. |
| `TRAIL_LENGTH` | `70` | Number of past analysis frames shown as the trailing "comet". |
| `FPS` | `30` | Target animation frame rate. |
| `ROTATE_SPEED` | `0.35` | Camera rotation speed in degrees per animation frame. |
| `BG_COLOR` | `#080810` | Background color (near-black with a hint of blue). |
| `GRID_COLOR` | `#14142a` | 3D grid/pane edge color. |
| `SENSITIVITY` | `2.0` | Gamma curve exponent (>1 amplifies subtle feature differences in color/position). |
| `PCT_LO` / `PCT_HI` | `2` / `98` | Percentiles used for contrast stretching (clips outliers so data fills the space). |

---

## Installation

### Prerequisites

- **Python 3.10+**
- **ffmpeg** (for MP4 export): `brew install ffmpeg`
- **portaudio** (for live audio playback): `brew install portaudio`

### Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading and spectral feature extraction |
| `numpy` | Numerical arrays and operations |
| `matplotlib` | 3D plotting and animation |
| `sounddevice` | Real-time audio playback (optional — the script runs silently without it) |

---

## Usage Examples

```bash
# Single file — live visualization with audio playback
python visualize.py recording.wav

# Single file — export as MP4 video with audio
python visualize.py recording.wav --export timbral_space.mp4

# Compare two files side by side
python compare.py bird1.wav bird2.wav
python compare.py bird1.wav bird2.wav --export comparison.mp4

# Real-time microphone visualization
python live.py
python live.py --duration 60 --window 20

# Works with any format librosa supports
python visualize.py birdsong.mp3
python visualize.py field_recording.flac
```

---

## Interpreting the Visualization

- **Tight clusters** mean the sound's timbre is stable (e.g., a sustained note).
- **Long, fast-moving trails** mean the timbre is changing rapidly (e.g., a bird's trill sweeping through frequencies).
- **High on the Z axis** = loud moments; **low** = quiet/silence.
- **Right on the X axis** = bright/high-frequency sounds; **left** = dark/low-frequency sounds.
- **High on the Y axis** = noisy/broadband; **low** = tonal/narrowband.
- **Vivid colors** = spectrally spread (noisy/rich); **muted colors** = spectrally focused (tonal/pure).
- **Bright dots** = loud moments; **dim dots** = quiet moments.
- **Hue shifts** track brightness — redder hues for darker sounds, bluer hues for brighter sounds.

For birdsong specifically, you'll typically see rapid jumps and spirals as the bird modulates pitch, volume, and harmonic content in quick succession — creating a complex, dynamic trajectory through timbral space.

In **compare mode**, you can see how two recordings differ: if one bird's trajectory occupies a different region or traces a different shape, that reveals concrete differences in their timbral character.
