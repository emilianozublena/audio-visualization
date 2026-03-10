#!/usr/bin/env python3
"""
Clean Timbral Space — no axes, no grid, just the animation in open space.

Same analysis and animation as visualize.py but with all chart chrome removed.

Usage:
  python clean.py <audio_file>
  python clean.py <audio_file> --export out.mp4
"""

import argparse
import os
import subprocess
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
HOP_LENGTH   = 512
N_FFT        = 2048
TRAIL_LENGTH = 70
FPS          = 30
ROTATE_SPEED = 0.35
BG_COLOR     = "#080810"
SENSITIVITY  = 2.0
PCT_LO       = 2
PCT_HI       = 98
# ─────────────────────────────────────────────────────────────────────────────


def _pick_backend():
    for backend in ("macosx", "TkAgg", "Qt5Agg", "GTK3Agg", "WebAgg"):
        try:
            matplotlib.use(backend)
            return backend
        except Exception:
            continue
    return None


_pick_backend()


def analyze(audio_path: str):
    print(f"\nLoading  : {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    print(f"Sample rate : {sr} Hz")
    print(f"Duration    : {len(y)/sr:.2f} s")
    print("Analyzing features …")

    kw = dict(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    centroid  = librosa.feature.spectral_centroid(**kw)[0]  / 1000.0
    bandwidth = librosa.feature.spectral_bandwidth(**kw)[0] / 1000.0
    rms       = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_db    = librosa.amplitude_to_db(rms, ref=np.max)

    n = min(len(centroid), len(bandwidth), len(rms_db))
    centroid, bandwidth, rms_db = centroid[:n], bandwidth[:n], rms_db[:n]
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)

    print("Done.\n")
    return y, sr, times, centroid, bandwidth, rms_db


def _sensitive_norm(arr):
    lo = np.percentile(arr, PCT_LO)
    hi = np.percentile(arr, PCT_HI)
    clipped = np.clip(arr, lo, hi)
    linear = (clipped - lo) / (hi - lo + 1e-8)
    return np.power(linear, 1.0 / SENSITIVITY)


def _stretch(arr):
    lo = np.percentile(arr, PCT_LO)
    hi = np.percentile(arr, PCT_HI)
    return np.clip(arr, lo, hi), lo, hi


def visualize_clean(audio_path: str, export_path: str | None = None):
    y, sr, times, centroid, bandwidth, rms_db = analyze(audio_path)
    n_frames = len(times)
    exporting = export_path is not None

    # Normalize for color
    norm_centroid  = _sensitive_norm(centroid)
    norm_bandwidth = _sensitive_norm(bandwidth)
    norm_rms       = _sensitive_norm(rms_db)

    hue = norm_centroid * 0.8
    sat = 0.3 + norm_bandwidth * 0.7
    val = 0.25 + norm_rms * 0.75
    hsv = np.stack([hue, sat, val], axis=-1)
    frame_colors = mcolors.hsv_to_rgb(hsv)

    # Stretch positions
    centroid_s,  cx_lo, cx_hi   = _stretch(centroid)
    bandwidth_s, bw_lo, bw_hi   = _stretch(bandwidth)
    rms_db_s,    rms_lo, rms_hi = _stretch(rms_db)

    if exporting:
        matplotlib.use("Agg")

    # ── FIGURE (no axes, no grid, no labels) ──────────────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)

    # Hide all axes chrome
    ax.set_axis_off()

    # Still need limits so the data is framed properly
    margin = 0.08
    ax.set_xlim(cx_lo  - margin, cx_hi  + margin)
    ax.set_ylim(bw_lo  - margin, bw_hi  + margin)
    ax.set_zlim(rms_lo - 1,      rms_hi + 1)
    ax.view_init(elev=22, azim=30)

    # ── DRAWABLE ELEMENTS ─────────────────────────────────────────────────
    trail_line, = ax.plot(
        [], [], [],
        color="#ffffff", alpha=0.12, linewidth=0.8, zorder=1
    )

    max_trail = TRAIL_LENGTH + 1
    scat = ax.scatter(
        np.zeros(max_trail), np.zeros(max_trail), np.zeros(max_trail),
        c=np.zeros((max_trail, 4)), s=np.zeros(max_trail), zorder=3
    )

    dot = ax.scatter(
        [centroid_s[0]], [bandwidth_s[0]], [rms_db_s[0]],
        c="white", s=110, alpha=1.0, zorder=5,
        edgecolors="#aaaacc", linewidths=1.2
    )

    # ── AUDIO ─────────────────────────────────────────────────────────────
    audio_available = False
    if not exporting:
        try:
            import sounddevice as sd
            audio_available = True
        except ImportError:
            print("sounddevice not found — running silent (pip install sounddevice)")

    audio_start = [None]

    if not exporting:
        def _start_audio(event):
            if audio_start[0] is None and audio_available:
                sd.play(y, sr)
                audio_start[0] = time.perf_counter()
        fig.canvas.mpl_connect("draw_event", _start_audio)

    # ── ANIMATION ─────────────────────────────────────────────────────────
    total_frames = int(times[-1] * FPS) + FPS

    def _frame_to_index(frame_num):
        t = frame_num / FPS
        return min(int(np.searchsorted(times, t)), n_frames - 1)

    def update(frame_num):
        if not exporting and audio_start[0] is not None:
            elapsed = time.perf_counter() - audio_start[0]
            i = min(int(np.searchsorted(times, elapsed)), n_frames - 1)
        else:
            i = _frame_to_index(frame_num)

        start = max(0, i - TRAIL_LENGTH)

        xs = centroid_s[start : i + 1]
        ys = bandwidth_s[start : i + 1]
        zs = rms_db_s[start : i + 1]
        cs = frame_colors[start : i + 1]
        n  = len(xs)

        trail_line.set_data(xs, ys)
        trail_line.set_3d_properties(zs)

        full_x = np.zeros(max_trail)
        full_y = np.zeros(max_trail)
        full_z = np.zeros(max_trail)
        full_rgba = np.zeros((max_trail, 4))
        full_sizes = np.zeros(max_trail)

        full_x[:n] = xs
        full_y[:n] = ys
        full_z[:n] = zs
        full_rgba[:n, :3] = cs
        full_rgba[:n, 3] = np.linspace(0.3, 0.9, n) if n > 1 else [0.9]
        full_sizes[:n] = np.linspace(6, 38, n) if n > 1 else [38]

        scat._offsets3d = (full_x, full_y, full_z)
        scat.set_facecolors(full_rgba)
        scat.set_sizes(full_sizes)

        dot._offsets3d = ([xs[-1]], [ys[-1]], [zs[-1]])
        ax.view_init(elev=22, azim=(30 + frame_num * ROTATE_SPEED) % 360)

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

    # Remove all padding so the 3D space fills the window
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if exporting:
        _export_mp4(anim, audio_path, export_path, total_frames)
    else:
        plt.show()


def _export_mp4(anim, audio_path, export_path, total_frames):
    fig = anim._fig
    dpi = 100
    fig.set_dpi(dpi)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

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

        if (frame_num + 1) % 30 == 0 or frame_num == total_frames - 1:
            pct = 100 * (frame_num + 1) / total_frames
            elapsed = time.perf_counter() - t0
            eta = elapsed / (frame_num + 1) * (total_frames - frame_num - 1)
            print(f"\r  {pct:5.1f}%  ({frame_num+1}/{total_frames})  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", end="", flush=True)

    proc.stdin.close()
    proc.wait()
    print()

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
    parser = argparse.ArgumentParser(description="Clean Timbral Space — no axes")
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument(
        "--export", metavar="OUT.mp4",
        help="Export animation as MP4 (requires ffmpeg)",
    )
    args = parser.parse_args()
    visualize_clean(args.audio, export_path=args.export)
