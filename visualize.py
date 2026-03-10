#!/usr/bin/env python3
"""
Timbral Space Visualizer
========================
Visualizes audio in a 3D timbral space, animated in sync with playback.

Axes:
  X  — Spectral Centroid (kHz)   : perceptual brightness of the sound
  Y  — Spectral Bandwidth (kHz)  : spectral spread around the centroid
  Z  — RMS Loudness (dBFS)       : overall energy / tonality level
  Color — HSV derived from all three features

Usage:
  python visualize.py <audio_file>                  # live window + audio
  python visualize.py <audio_file> --export out.mp4  # save MP4 with audio

Supported formats: WAV, MP3, FLAC, OGG, AIFF, and anything librosa can open.
Requires ffmpeg on PATH for MP4 export.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import librosa
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOP_LENGTH   = 512       # Samples between each analysis frame (smaller = more detail, slower)
N_FFT        = 2048      # Size of the FFT window (controls frequency resolution)
TRAIL_LENGTH = 70        # How many past points the "tail" behind the dot shows
FPS          = 30        # Animation framerate
ROTATE_SPEED = 0.35      # How fast the 3D camera rotates (degrees per frame)
BG_COLOR     = "#080810" # Dark background for the plot
GRID_COLOR   = "#14142a"
LABEL_COLOR  = "#888899"
TICK_COLOR   = "#444455"
SENSITIVITY  = 2.0       # > 1 amplifies subtle differences in the features (gamma curve)
PCT_LO       = 2         # Low percentile for contrast stretching (clips outliers)
PCT_HI       = 98        # High percentile for contrast stretching
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


def analyze(audio_path: str):
    """Load audio and extract frame-by-frame timbral features."""
    print(f"\nLoading  : {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"Sample rate : {sr} Hz")
    print(f"Duration    : {duration:.2f} s")
    print("Analyzing features …")

    kw = dict(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Extract the three timbral features for every analysis frame:
    # - Spectral centroid: "brightness" — where the center of mass of the spectrum is (in kHz)
    # - Spectral bandwidth: "texture" — how spread out the frequencies are around the centroid (in kHz)
    # - RMS energy → converted to dB: "loudness" — overall volume level at each moment
    centroid  = librosa.feature.spectral_centroid(**kw)[0]  / 1000.0   # Hz → kHz
    bandwidth = librosa.feature.spectral_bandwidth(**kw)[0] / 1000.0   # Hz → kHz
    rms       = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_db    = librosa.amplitude_to_db(rms, ref=np.max)               # amplitude → decibels

    # Trim all arrays to the same length (they can differ by ±1 frame)
    n = min(len(centroid), len(bandwidth), len(rms_db))
    centroid, bandwidth, rms_db = centroid[:n], bandwidth[:n], rms_db[:n]

    # Convert frame indices to timestamps (seconds)
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)

    print(f"Centroid    : {centroid.min():.2f} – {centroid.max():.2f} kHz")
    print(f"Bandwidth   : {bandwidth.min():.2f} – {bandwidth.max():.2f} kHz")
    print(f"Loudness    : {rms_db.min():.1f} – {rms_db.max():.1f} dBFS")
    print("Done.\n")

    return y, sr, times, centroid, bandwidth, rms_db


def _sensitive_norm(arr):
    """Normalize an array to 0–1 using percentile clipping + gamma curve.

    This avoids outliers from dominating the range and applies a power curve
    (controlled by SENSITIVITY) so subtle variations become more visible.
    """
    lo = np.percentile(arr, PCT_LO)
    hi = np.percentile(arr, PCT_HI)
    clipped = np.clip(arr, lo, hi)
    linear = (clipped - lo) / (hi - lo + 1e-8)
    return np.power(linear, 1.0 / SENSITIVITY)


def _stretch(arr):
    """Contrast-stretch an array by clipping to its 2nd–98th percentile range.

    Returns the clipped array plus the bounds (used for setting axis limits).
    """
    lo = np.percentile(arr, PCT_LO)
    hi = np.percentile(arr, PCT_HI)
    return np.clip(arr, lo, hi), lo, hi


def _style_ax(ax):
    """Apply the dark theme to a 3D axes (background, grid, tick colors)."""
    ax.set_facecolor(BG_COLOR)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(LABEL_COLOR)
        axis._axinfo["tick"]["color"] = TICK_COLOR
    ax.tick_params(colors=TICK_COLOR, labelsize=8)


def visualize(audio_path: str, export_path: str | None = None):
    y, sr, times, centroid, bandwidth, rms_db = analyze(audio_path)
    n_frames = len(times)
    exporting = export_path is not None

    # ── NORMALIZE FEATURES FOR COLOR MAPPING ─────────────────────────────────
    # Each feature is normalized to 0–1 (with gamma) so we can map it to a color channel
    norm_centroid  = _sensitive_norm(centroid)
    norm_bandwidth = _sensitive_norm(bandwidth)
    norm_rms       = _sensitive_norm(rms_db)

    # ── MAP FEATURES → HSV COLOR ─────────────────────────────────────────────
    # Centroid  → Hue (bright sounds are bluer, dark sounds are redder)
    # Bandwidth → Saturation (spread-out spectrum = more vivid color)
    # Loudness  → Value/brightness (louder = brighter dot)
    hue = norm_centroid * 0.8
    sat = 0.3 + norm_bandwidth * 0.7
    val = 0.25 + norm_rms * 0.75

    hsv = np.stack([hue, sat, val], axis=-1)
    frame_colors = mcolors.hsv_to_rgb(hsv)     # convert HSV → RGB for matplotlib

    # ── CONTRAST-STRETCH POSITIONS FOR 3D AXES ───────────────────────────────
    # Clip outliers so the point cloud fills the 3D space nicely
    centroid_s,  cx_lo, cx_hi  = _stretch(centroid)
    bandwidth_s, bw_lo, bw_hi = _stretch(bandwidth)
    rms_db_s,    rms_lo, rms_hi = _stretch(rms_db)

    if exporting:
        matplotlib.use("Agg")

    # ── FIGURE ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d")
    _style_ax(ax)

    margin = 0.08
    ax.set_xlim(cx_lo  - margin, cx_hi  + margin)
    ax.set_ylim(bw_lo  - margin, bw_hi  + margin)
    ax.set_zlim(rms_lo - 1,      rms_hi + 1)

    ax.set_xlabel("Centroid (kHz)",  labelpad=12)
    ax.set_ylabel("Bandwidth (kHz)", labelpad=12)
    ax.set_zlabel("Loudness (dBFS)", labelpad=8)
    ax.view_init(elev=22, azim=30)

    fig.suptitle(
        "Timbral Space",
        color="#ccccdd", fontsize=15, fontweight="light", y=0.97
    )
    time_text = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes,
        color="#555566", fontsize=9
    )

    # Colorbar
    cbar_cmap = mcolors.LinearSegmentedColormap.from_list(
        "centroid_hue",
        [mcolors.hsv_to_rgb((h, 0.8, 0.9)) for h in np.linspace(0, 0.8, 256)],
    )
    sm = plt.cm.ScalarMappable(
        cmap=cbar_cmap,
        norm=plt.Normalize(vmin=cx_lo, vmax=cx_hi),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.45, aspect=20)
    cbar.set_label("Centroid / Hue (kHz)", color=LABEL_COLOR, fontsize=9)
    cbar.ax.tick_params(colors=TICK_COLOR, labelsize=8)
    cbar.outline.set_edgecolor(GRID_COLOR)

    # ── DRAWABLE ELEMENTS ("artists" in matplotlib) ─────────────────────────
    # Thin line connecting the recent trail points
    trail_line, = ax.plot(
        [], [], [],
        color="#ffffff", alpha=0.12, linewidth=0.8, zorder=1
    )

    # Scatter plot for the fading trail — pre-allocated at max size and updated
    # in-place each frame for performance (avoids creating new objects every frame)
    max_trail = TRAIL_LENGTH + 1
    scat = ax.scatter(
        np.zeros(max_trail), np.zeros(max_trail), np.zeros(max_trail),
        c=np.zeros((max_trail, 4)), s=np.zeros(max_trail), zorder=3
    )

    # The bright "current position" dot at the head of the trail
    dot = ax.scatter(
        [centroid_s[0]], [bandwidth_s[0]], [rms_db_s[0]],
        c="white", s=110, alpha=1.0, zorder=5,
        edgecolors="#aaaacc", linewidths=1.2
    )

    # ── AUDIO PLAYBACK (only when showing the live window, not when exporting) ─
    audio_available = False
    if not exporting:
        try:
            import sounddevice as sd
            audio_available = True
        except ImportError:
            print("sounddevice not found — running silent (pip install sounddevice)")

    # We use a list so the nested function below can modify it (closure trick)
    audio_start = [None]

    if not exporting:
        def _start_audio(event):
            """Begin audio playback the first time the window draws."""
            if audio_start[0] is None and audio_available:
                sd.play(y, sr)
                audio_start[0] = time.perf_counter()
        fig.canvas.mpl_connect("draw_event", _start_audio)

    # ── ANIMATION ─────────────────────────────────────────────────────────────
    total_frames = int(times[-1] * FPS) + FPS

    def _frame_to_index(frame_num):
        t = frame_num / FPS
        return min(int(np.searchsorted(times, t)), n_frames - 1)

    def update(frame_num):
        """Called every frame — moves the dot and trail to the current position."""
        # In live mode, sync to real elapsed time (matches audio playback).
        # In export mode, derive time from the frame number.
        if not exporting and audio_start[0] is not None:
            elapsed = time.perf_counter() - audio_start[0]
            i = min(int(np.searchsorted(times, elapsed)), n_frames - 1)
        else:
            i = _frame_to_index(frame_num)

        # Grab the last TRAIL_LENGTH points up to the current position
        start = max(0, i - TRAIL_LENGTH)

        xs = centroid_s[start : i + 1]    # X positions (brightness)
        ys = bandwidth_s[start : i + 1]   # Y positions (texture)
        zs = rms_db_s[start : i + 1]      # Z positions (loudness)
        cs = frame_colors[start : i + 1]  # RGB colors for each point
        n  = len(xs)

        # Update the thin connecting line
        trail_line.set_data(xs, ys)
        trail_line.set_3d_properties(zs)

        # Fill the pre-allocated scatter arrays (zeros beyond the trail are invisible)
        full_x = np.zeros(max_trail)
        full_y = np.zeros(max_trail)
        full_z = np.zeros(max_trail)
        full_rgba = np.zeros((max_trail, 4))
        full_sizes = np.zeros(max_trail)

        full_x[:n] = xs
        full_y[:n] = ys
        full_z[:n] = zs
        full_rgba[:n, :3] = cs
        # Trail fades in: older points are more transparent, newer points more opaque
        full_rgba[:n, 3] = np.linspace(0.3, 0.9, n) if n > 1 else [0.9]
        # Trail grows: older points are smaller, the newest is largest
        full_sizes[:n] = np.linspace(6, 38, n) if n > 1 else [38]

        scat._offsets3d = (full_x, full_y, full_z)
        scat.set_facecolors(full_rgba)
        scat.set_sizes(full_sizes)

        # Move the bright head dot to the current position
        dot._offsets3d = ([xs[-1]], [ys[-1]], [zs[-1]])
        # Slowly rotate the camera for a dynamic 3D feel
        ax.view_init(elev=22, azim=(30 + frame_num * ROTATE_SPEED) % 360)
        time_text.set_text(f"t = {times[i]:.2f} s")

        # Stop when the audio ends
        if not exporting and i >= n_frames - 1:
            anim.event_source.stop()
            if audio_available:
                sd.stop()

        return trail_line, scat, dot

    anim = FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=False
    )

    plt.tight_layout()

    if exporting:
        _export_mp4(anim, audio_path, export_path, total_frames)
    else:
        plt.show()


def _export_mp4(anim, audio_path, export_path, total_frames):
    """Render the animation to an MP4 file with audio.

    Pipeline: render raw RGBA frames → pipe to ffmpeg → mux original audio on top.
    This is faster than matplotlib's built-in anim.save().
    """
    fig = anim._fig
    dpi = 100
    fig.set_dpi(dpi)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Create a temporary video file (no audio yet)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgba",
                "-r", str(FPS),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "20",
                tmp_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("ffmpeg not found — cannot export.")
        return

    print(f"Rendering {total_frames} frames …")
    t0 = time.perf_counter()
    for frame_num in range(total_frames):
        anim._func(frame_num)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        proc.stdin.write(bytes(buf))

        # Progress
        if (frame_num + 1) % 30 == 0 or frame_num == total_frames - 1:
            pct = 100 * (frame_num + 1) / total_frames
            elapsed = time.perf_counter() - t0
            eta = elapsed / (frame_num + 1) * (total_frames - frame_num - 1)
            print(f"\r  {pct:5.1f}%  ({frame_num+1}/{total_frames})  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", end="", flush=True)

    proc.stdin.close()
    proc.wait()
    print()

    # Mux audio
    print("Muxing audio …")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                export_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"Exported → {export_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg mux failed: {e.stderr[:200]}")
        os.rename(tmp_path, export_path)
        print(f"Exported (silent) → {export_path}")
        return
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timbral Space Visualizer")
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument(
        "--export", metavar="OUT.mp4",
        help="Export animation as MP4 (requires ffmpeg)",
    )
    args = parser.parse_args()
    visualize(args.audio, export_path=args.export)
