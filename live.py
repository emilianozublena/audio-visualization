#!/usr/bin/env python3
"""
Live Timbral Space — Real-time microphone visualization.

Captures audio from the default input device and animates the timbral
trajectory in real time. The rolling window shows the last ~10 seconds
of sound, with axes that auto-scale to the incoming signal.

Usage:
  python live.py
  python live.py --duration 30        # stop after 30 seconds
  python live.py --window 15          # 15-second rolling window
"""

import argparse
import queue
import time
import warnings
from collections import deque

import numpy as np
import librosa
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050   # Audio capture sample rate (lower = less CPU, good enough for analysis)
BLOCK_SIZE     = 2048    # Samples per audio callback (how often we get new audio data)
HOP_LENGTH     = 512     # Samples between analysis frames
N_FFT          = 2048    # FFT window size (frequency resolution)
TRAIL_LENGTH   = 70      # Number of past points shown in the trail
FPS            = 30      # Animation framerate
ROTATE_SPEED   = 0.35    # Camera rotation speed (degrees per frame)
BG_COLOR       = "#080810"
GRID_COLOR     = "#14142a"
LABEL_COLOR    = "#888899"
TICK_COLOR     = "#444455"
SENSITIVITY    = 2.0     # Gamma curve — amplifies subtle feature differences
WINDOW_SEC     = 10.0    # Rolling window: only keep the last N seconds of data
# ─────────────────────────────────────────────────────────────────────────────


def _pick_backend():
    """Try each matplotlib GUI backend until one works on this OS."""
    for backend in ("macosx", "TkAgg", "Qt5Agg", "GTK3Agg", "WebAgg"):
        try:
            matplotlib.use(backend)
            return backend
        except Exception:
            continue
    return None


_pick_backend()


def _style_ax(ax):
    """Apply the dark theme to a 3D axes."""
    ax.set_facecolor(BG_COLOR)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(LABEL_COLOR)
        axis._axinfo["tick"]["color"] = TICK_COLOR
    ax.tick_params(colors=TICK_COLOR, labelsize=8)


def live(duration: float | None = None, window_sec: float = WINDOW_SEC):
    """Capture audio from the microphone and animate the timbral space in real time.

    Unlike visualize.py (which pre-analyzes a file), this processes audio on-the-fly:
    a background thread captures mic input, and the animation loop analyzes it frame by frame.
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is required for live mode: pip install sounddevice")
        return

    # Thread-safe queue: the audio callback (runs in a separate thread) pushes
    # chunks here, and the animation loop pulls them out for analysis
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice whenever a new block of audio is available."""
        audio_q.put(indata[:, 0].copy())

    # Rolling storage for computed features — old data falls off automatically
    max_points = int(window_sec * SAMPLE_RATE / HOP_LENGTH)
    centroids = deque(maxlen=max_points)
    bandwidths = deque(maxlen=max_points)
    rms_vals = deque(maxlen=max_points)

    # Axis limits adapt smoothly over time using exponential moving average (EMA).
    # This prevents the axes from jumping wildly when a sudden loud/bright sound appears.
    ema_alpha = 0.02
    running = {
        "cx_lo": 0.5, "cx_hi": 4.0,
        "bw_lo": 0.5, "bw_hi": 4.0,
        "rms_lo": -60.0, "rms_hi": -10.0,
    }

    # Accumulate incoming audio samples until we have enough for one FFT window
    audio_buf = np.zeros(0, dtype=np.float32)

    # ── FIGURE ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    _style_ax(ax)

    ax.set_xlabel("Centroid (kHz)", labelpad=12)
    ax.set_ylabel("Bandwidth (kHz)", labelpad=12)
    ax.set_zlabel("Loudness (dBFS)", labelpad=8)
    ax.view_init(elev=22, azim=30)

    fig.suptitle(
        "Live Timbral Space",
        color="#ccccdd", fontsize=15, fontweight="light", y=0.97
    )
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color="#555566", fontsize=9)
    live_indicator = ax.text2D(0.92, 0.95, "● LIVE", transform=ax.transAxes,
                               color="#ff4444", fontsize=11, fontweight="bold")

    # ── DRAWABLE ELEMENTS ──────────────────────────────────────────────────────
    # Thin line connecting trail points
    trail_line, = ax.plot([], [], [], color="#ffffff", alpha=0.12, linewidth=0.8, zorder=1)

    # Fading scatter trail (pre-allocated, updated in-place each frame)
    max_trail = TRAIL_LENGTH + 1
    scat = ax.scatter(
        np.zeros(max_trail), np.zeros(max_trail), np.zeros(max_trail),
        c=np.zeros((max_trail, 4)), s=np.zeros(max_trail), zorder=3
    )
    # Bright head dot showing the current sound position
    dot = ax.scatter(
        [0], [0], [0], c="white", s=110, alpha=1.0, zorder=5,
        edgecolors="#aaaacc", linewidths=1.2
    )

    start_time = time.perf_counter()

    def update(frame_num):
        """Called every frame — pulls new audio, analyzes it, and updates the plot."""
        nonlocal audio_buf

        # Pull all available audio chunks from the capture thread
        while not audio_q.empty():
            chunk = audio_q.get_nowait()
            audio_buf = np.concatenate([audio_buf, chunk])

        # Analyze in sliding windows: each time we have enough samples for one FFT,
        # extract features and slide forward by HOP_LENGTH samples
        min_samples = N_FFT
        while len(audio_buf) >= min_samples:
            block = audio_buf[:min_samples]
            audio_buf = audio_buf[HOP_LENGTH:]

            # Same three features as the file-based visualizer:
            # centroid (brightness), bandwidth (texture), RMS (loudness)
            kw = dict(y=block, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
            c = librosa.feature.spectral_centroid(**kw)[0] / 1000.0
            b = librosa.feature.spectral_bandwidth(**kw)[0] / 1000.0
            r = librosa.feature.rms(y=block, hop_length=HOP_LENGTH)[0]
            r_db = librosa.amplitude_to_db(r, ref=1.0)

            for val_c, val_b, val_r in zip(c, b, r_db):
                centroids.append(val_c)
                bandwidths.append(val_b)
                rms_vals.append(val_r)

        n = len(centroids)
        if n < 2:
            return trail_line, scat, dot

        cx = np.array(centroids)
        bw = np.array(bandwidths)
        rm = np.array(rms_vals)

        # Smoothly adapt axis limits to the incoming data (EMA avoids sudden jumps)
        running["cx_lo"] += ema_alpha * (cx.min() - running["cx_lo"])
        running["cx_hi"] += ema_alpha * (cx.max() - running["cx_hi"])
        running["bw_lo"] += ema_alpha * (bw.min() - running["bw_lo"])
        running["bw_hi"] += ema_alpha * (bw.max() - running["bw_hi"])
        running["rms_lo"] += ema_alpha * (rm.min() - running["rms_lo"])
        running["rms_hi"] += ema_alpha * (rm.max() - running["rms_hi"])

        margin = 0.08
        ax.set_xlim(running["cx_lo"] - margin, running["cx_hi"] + margin)
        ax.set_ylim(running["bw_lo"] - margin, running["bw_hi"] + margin)
        ax.set_zlim(running["rms_lo"] - 1, running["rms_hi"] + 1)

        # Normalize features against the running range for color mapping
        def _live_norm(arr, lo, hi):
            """Same gamma-curved normalization, but using live running bounds."""
            rng = hi - lo + 1e-8
            linear = np.clip((arr - lo) / rng, 0, 1)
            return np.power(linear, 1.0 / SENSITIVITY)

        norm_c = _live_norm(cx, running["cx_lo"], running["cx_hi"])
        norm_b = _live_norm(bw, running["bw_lo"], running["bw_hi"])
        norm_r = _live_norm(rm, running["rms_lo"], running["rms_hi"])

        # Map features → color (same logic as visualize.py)
        # Centroid → hue, bandwidth → saturation, loudness → brightness
        hue = norm_c * 0.8
        sat = 0.3 + norm_b * 0.7
        val = 0.25 + norm_r * 0.75
        hsv = np.stack([hue, sat, val], axis=-1)
        colors = mcolors.hsv_to_rgb(hsv)

        # Show only the most recent TRAIL_LENGTH points
        trail_start = max(0, n - TRAIL_LENGTH)
        xs = cx[trail_start:]
        ys = bw[trail_start:]
        zs = rm[trail_start:]
        cs = colors[trail_start:]
        tn = len(xs)

        trail_line.set_data(xs, ys)
        trail_line.set_3d_properties(zs)

        full_x = np.zeros(max_trail)
        full_y = np.zeros(max_trail)
        full_z = np.zeros(max_trail)
        full_rgba = np.zeros((max_trail, 4))
        full_sizes = np.zeros(max_trail)

        full_x[:tn] = xs
        full_y[:tn] = ys
        full_z[:tn] = zs
        full_rgba[:tn, :3] = cs
        full_rgba[:tn, 3] = np.linspace(0.3, 0.9, tn) if tn > 1 else [0.9]
        full_sizes[:tn] = np.linspace(6, 38, tn) if tn > 1 else [38]

        scat._offsets3d = (full_x, full_y, full_z)
        scat.set_facecolors(full_rgba)
        scat.set_sizes(full_sizes)

        dot._offsets3d = ([xs[-1]], [ys[-1]], [zs[-1]])

        ax.view_init(elev=22, azim=(30 + frame_num * ROTATE_SPEED) % 360)
        elapsed = time.perf_counter() - start_time
        time_text.set_text(f"t = {elapsed:.1f} s  |  {n} pts")

        # Pulse the "LIVE" indicator to show the system is active
        live_indicator.set_alpha(0.5 + 0.5 * np.sin(elapsed * 3))

        if duration is not None and elapsed >= duration:
            anim.event_source.stop()

        return trail_line, scat, dot

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )

    with stream:
        print(f"Listening on default input device …")
        print(f"Rolling window: {window_sec}s | Press Ctrl+C or close window to stop.\n")

        anim = FuncAnimation(
            fig, update,
            interval=1000 / FPS,
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()

    print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Timbral Space — microphone mode")
    parser.add_argument("--duration", type=float, default=None,
                        help="Auto-stop after N seconds")
    parser.add_argument("--window", type=float, default=WINDOW_SEC,
                        help=f"Rolling window in seconds (default {WINDOW_SEC})")
    args = parser.parse_args()
    live(duration=args.duration, window_sec=args.window)
