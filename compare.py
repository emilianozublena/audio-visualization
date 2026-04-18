#!/usr/bin/env python3
"""
Timbral Space Comparator
========================
Visualizes TWO audio files side-by-side in a shared 3D timbral space,
each with its own distinct color palette.

  Track A (first file)  → cyan / blue palette
  Track B (second file) → orange / red palette

Usage:
  python compare.py bird1.wav bird2.wav
  python compare.py bird1.wav bird2.wav --export comparison.mp4

Both tracks are mixed for audio playback / MP4 export.
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
HOP_LENGTH   = 512       # Samples between each analysis frame
N_FFT        = 2048      # FFT window size (frequency resolution)
TRAIL_LENGTH = 70        # Number of past points in each track's trail
FPS          = 30        # Animation framerate (live playback)
EXPORT_FPS   = 15        # Lower FPS for export — halves render time
ROTATE_SPEED = 0.35      # Camera rotation speed (degrees per frame)
BG_COLOR     = "#080810" # Dark background
GRID_COLOR   = "#14142a"
LABEL_COLOR  = "#888899"
TICK_COLOR   = "#444455"
SENSITIVITY  = 2.0       # Gamma curve — amplifies subtle feature differences
PCT_LO       = 2         # Low percentile for contrast stretching
PCT_HI       = 98        # High percentile for contrast stretching

# Each track gets its own color palette so they're visually distinct.
# Format: (hue_center, hue_half_width) — defines the HSV hue range for each track.
PALETTES = [
    (0.55, 0.15),   # Track A: cyan → blue
    (0.08, 0.08),   # Track B: red → orange
]
TRAIL_COLORS = ["#44aacc", "#cc6633"]   # Thin trail line color per track
DOT_COLORS   = ["#88ddff", "#ffaa55"]   # Bright head dot color per track
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


def _sensitive_norm(arr):
    """Normalize to 0–1 with percentile clipping + gamma curve.

    Clips outliers and applies a power curve so subtle variations pop out.
    """
    lo = np.percentile(arr, PCT_LO)
    hi = np.percentile(arr, PCT_HI)
    clipped = np.clip(arr, lo, hi)
    linear = (clipped - lo) / (hi - lo + 1e-8)
    return np.power(linear, 1.0 / SENSITIVITY)


def _stretch(arr):
    """Contrast-stretch: clip to 2nd–98th percentile, return clipped array + bounds."""
    lo = np.percentile(arr, PCT_LO)
    hi = np.percentile(arr, PCT_HI)
    return np.clip(arr, lo, hi), lo, hi


def analyze(audio_path: str):
    print(f"\nLoading  : {audio_path}")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    print(f"Sample rate : {sr} Hz")
    print(f"Duration    : {len(y)/sr:.2f} s")
    print("Analyzing features …")

    kw = dict(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Extract the three timbral features for every analysis frame:
    # - Spectral centroid: "brightness" — center of mass of the frequency spectrum (kHz)
    # - Spectral bandwidth: "texture" — how spread out the frequencies are (kHz)
    # - RMS → dB: "loudness" — overall energy level at each moment
    centroid  = librosa.feature.spectral_centroid(**kw)[0]  / 1000.0   # Hz → kHz
    bandwidth = librosa.feature.spectral_bandwidth(**kw)[0] / 1000.0   # Hz → kHz
    rms       = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_db    = librosa.amplitude_to_db(rms, ref=np.max)               # amplitude → decibels

    # Trim to equal length and convert frame indices to timestamps
    n = min(len(centroid), len(bandwidth), len(rms_db))
    centroid, bandwidth, rms_db = centroid[:n], bandwidth[:n], rms_db[:n]
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)

    print(f"Centroid    : {centroid.min():.2f} – {centroid.max():.2f} kHz")
    print(f"Bandwidth   : {bandwidth.min():.2f} – {bandwidth.max():.2f} kHz")
    print(f"Loudness    : {rms_db.min():.1f} – {rms_db.max():.1f} dBFS")
    print("Done.\n")

    return y, sr, times, centroid, bandwidth, rms_db


def _build_track(audio_path, palette_idx):
    """Analyze one audio file and prepare everything needed to animate it.

    Returns a dict with positions, colors, and metadata for the track.
    Each track gets its own color palette (cyan/blue vs orange/red) so
    you can tell them apart when they share the same 3D space.
    """
    y, sr, times, centroid, bandwidth, rms_db = analyze(audio_path)

    # Normalize features to 0–1 for color mapping
    norm_c = _sensitive_norm(centroid)
    norm_b = _sensitive_norm(bandwidth)
    norm_r = _sensitive_norm(rms_db)

    # Map features → HSV color, restricted to this track's hue range
    # Centroid shifts the hue within the palette, bandwidth → saturation, loudness → brightness
    hue_center, hue_hw = PALETTES[palette_idx]
    hue = hue_center + (norm_c - 0.5) * hue_hw * 2
    hue = np.clip(hue, 0.0, 1.0)
    sat = 0.3 + norm_b * 0.7
    val = 0.25 + norm_r * 0.75

    hsv = np.stack([hue, sat, val], axis=-1)
    colors = mcolors.hsv_to_rgb(hsv)

    c_s, cx_lo, cx_hi   = _stretch(centroid)
    b_s, bw_lo, bw_hi   = _stretch(bandwidth)
    r_s, rms_lo, rms_hi = _stretch(rms_db)

    return dict(
        y=y, sr=sr, times=times, n_frames=len(times),
        centroid_s=c_s, bandwidth_s=b_s, rms_db_s=r_s,
        colors=colors, palette_idx=palette_idx,
        bounds=dict(cx=(cx_lo, cx_hi), bw=(bw_lo, bw_hi), rms=(rms_lo, rms_hi)),
        label=os.path.basename(audio_path),
    )


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


def _mix_audio(tracks):
    """Mix both audio tracks into a single array for simultaneous playback.

    If the two files have different sample rates, resample to the higher one.
    The result is normalized so it doesn't clip.
    """
    target_sr = max(t["sr"] for t in tracks)
    arrays = []
    for t in tracks:
        audio = t["y"]
        if t["sr"] != target_sr:
            audio = librosa.resample(audio, orig_sr=t["sr"], target_sr=target_sr)
        arrays.append(audio)

    max_len = max(len(a) for a in arrays)
    mixed = np.zeros(max_len, dtype=np.float32)
    for a in arrays:
        mixed[:len(a)] += a

    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak

    return mixed, target_sr


def compare(audio_paths: list[str], export_path: str | None = None):
    tracks = [_build_track(p, i) for i, p in enumerate(audio_paths)]
    exporting = export_path is not None

    # ── SHARED AXIS LIMITS ────────────────────────────────────────────────────
    # Use the combined min/max across both tracks so they share the same scale
    cx_lo  = min(t["bounds"]["cx"][0]  for t in tracks)
    cx_hi  = max(t["bounds"]["cx"][1]  for t in tracks)
    bw_lo  = min(t["bounds"]["bw"][0]  for t in tracks)
    bw_hi  = max(t["bounds"]["bw"][1]  for t in tracks)
    rms_lo = min(t["bounds"]["rms"][0] for t in tracks)
    rms_hi = max(t["bounds"]["rms"][1] for t in tracks)

    max_time = max(t["times"][-1] for t in tracks)

    fps = EXPORT_FPS if exporting else FPS

    if exporting:
        matplotlib.use("Agg")

    # ── FIGURE ────────────────────────────────────────────────────────────────
    fig_size = (8, 4.5) if exporting else (13, 9)
    fig = plt.figure(figsize=fig_size, facecolor=BG_COLOR)
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

    title = " vs ".join(t["label"] for t in tracks)
    fig.suptitle(
        f"Timbral Space — {title}",
        color="#ccccdd", fontsize=14, fontweight="light", y=0.97
    )
    time_text = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes,
        color="#555566", fontsize=9
    )

    # ── LEGEND ────────────────────────────────────────────────────────────────
    for idx, t in enumerate(tracks):
        ax.text2D(
            0.02, 0.90 - idx * 0.04, f"● {t['label']}",
            transform=ax.transAxes,
            color=DOT_COLORS[idx], fontsize=10, fontweight="bold",
        )

    # ── PER-TRACK DRAWABLE ELEMENTS ─────────────────────────────────────────
    # Each track gets its own trail line, scatter trail, and head dot
    trail_lines = []
    scats = []
    dots = []

    for idx, t in enumerate(tracks):
        # Thin connecting line for the trail
        line, = ax.plot(
            [], [], [],
            color=TRAIL_COLORS[idx], alpha=0.15, linewidth=0.8, zorder=1
        )
        trail_lines.append(line)

        # Fading scatter trail (pre-allocated for performance)
        max_trail = TRAIL_LENGTH + 1
        s = ax.scatter(
            np.zeros(max_trail), np.zeros(max_trail), np.zeros(max_trail),
            c=np.zeros((max_trail, 4)), s=np.zeros(max_trail), zorder=3 + idx
        )
        scats.append(s)

        # Bright "current position" dot
        d = ax.scatter(
            [t["centroid_s"][0]], [t["bandwidth_s"][0]], [t["rms_db_s"][0]],
            c=DOT_COLORS[idx], s=110, alpha=1.0, zorder=5 + idx,
            edgecolors=DOT_COLORS[idx], linewidths=1.2
        )
        dots.append(d)

    # ── AUDIO (live mode only) ────────────────────────────────────────────────
    audio_available = False
    mixed_y, mixed_sr = None, None
    if not exporting:
        try:
            import sounddevice as sd
            audio_available = True
            mixed_y, mixed_sr = _mix_audio(tracks)
        except ImportError:
            print("sounddevice not found — running silent (pip install sounddevice)")

    audio_start = [None]

    if not exporting:
        def _start_audio(event):
            if audio_start[0] is None and audio_available:
                sd.play(mixed_y, mixed_sr)
                audio_start[0] = time.perf_counter()
        fig.canvas.mpl_connect("draw_event", _start_audio)

    # ── PRE-COMPUTE FRAME DATA (export only) ────────────────────────────────
    # Avoid per-frame numpy allocations by pre-computing trail arrays
    if exporting:
        _precomputed = []
        for t in tracks:
            mt = TRAIL_LENGTH + 1
            nf = t["n_frames"]
            # Pre-compute the frame index for each animation frame
            frame_count = int(max_time * fps) + fps
            indices = np.minimum(
                np.searchsorted(t["times"], np.arange(frame_count) / fps).astype(int),
                nf - 1
            )
            _precomputed.append(dict(indices=indices, mt=mt))

    # ── ANIMATION ─────────────────────────────────────────────────────────────
    total_frames = int(max_time * fps) + fps

    def update(frame_num):
        """Called every frame — updates both tracks' dots and trails."""
        # Sync to real time during playback, or derive from frame number when exporting
        if not exporting and audio_start[0] is not None:
            elapsed = time.perf_counter() - audio_start[0]
        else:
            elapsed = frame_num / fps

        # Update each track independently
        for idx, t in enumerate(tracks):
            nf = t["n_frames"]
            if exporting:
                i = _precomputed[idx]["indices"][frame_num]
            else:
                i = min(int(np.searchsorted(t["times"], elapsed)), nf - 1)
            start = max(0, i - TRAIL_LENGTH)

            xs = t["centroid_s"][start : i + 1]
            ys = t["bandwidth_s"][start : i + 1]
            zs = t["rms_db_s"][start : i + 1]
            cs = t["colors"][start : i + 1]
            n  = len(xs)

            trail_lines[idx].set_data(xs, ys)
            trail_lines[idx].set_3d_properties(zs)

            # Update scatter in-place
            mt = TRAIL_LENGTH + 1
            full_x = np.zeros(mt)
            full_y = np.zeros(mt)
            full_z = np.zeros(mt)
            full_rgba = np.zeros((mt, 4))
            full_sizes = np.zeros(mt)

            full_x[:n] = xs
            full_y[:n] = ys
            full_z[:n] = zs
            full_rgba[:n, :3] = cs
            full_rgba[:n, 3] = np.linspace(0.3, 0.9, n) if n > 1 else [0.9]
            full_sizes[:n] = np.linspace(6, 38, n) if n > 1 else [38]

            scats[idx]._offsets3d = (full_x, full_y, full_z)
            scats[idx].set_facecolors(full_rgba)
            scats[idx].set_sizes(full_sizes)

            dots[idx]._offsets3d = ([xs[-1]], [ys[-1]], [zs[-1]])

        ax.view_init(elev=22, azim=(30 + frame_num * ROTATE_SPEED) % 360)
        time_text.set_text(f"t = {elapsed:.2f} s")

        all_done = all(elapsed >= t["times"][-1] for t in tracks)
        if not exporting and all_done:
            anim.event_source.stop()
            if audio_available:
                sd.stop()

        return trail_lines + scats + dots

    anim = FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=1000 / fps,
        blit=False
    )

    plt.tight_layout()

    if exporting:
        _export_mp4(anim, audio_paths, export_path, total_frames)
    else:
        plt.show()


def _export_mp4(anim, audio_paths, export_path, total_frames):
    """Render animation to MP4: pipe raw frames to ffmpeg, then mux both audio tracks."""
    fig = anim._fig
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
                "-r", str(EXPORT_FPS),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "23",
                tmp_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("ffmpeg not found — cannot export.")
        return

    print(f"Rendering {total_frames} frames @ {EXPORT_FPS} FPS …")
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

    # Mux audio — mix both tracks via ffmpeg amix filter
    print("Muxing audio …")
    cmd = ["ffmpeg", "-y", "-i", tmp_path]
    for p in audio_paths:
        cmd += ["-i", p]

    inputs = "".join(f"[{i+1}:a]" for i in range(len(audio_paths)))
    cmd += [
        "-filter_complex",
        f"{inputs}amix=inputs={len(audio_paths)}:duration=longest",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        export_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
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
    parser = argparse.ArgumentParser(description="Timbral Space Comparator")
    parser.add_argument(
        "audio", nargs=2, metavar="AUDIO",
        help="Paths to two audio files to compare",
    )
    parser.add_argument(
        "--export", metavar="OUT.mp4",
        help="Export animation as MP4 (requires ffmpeg)",
    )
    args = parser.parse_args()
    compare(args.audio, export_path=args.export)
