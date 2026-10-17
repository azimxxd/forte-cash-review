"""Find short time windows with fine hand/desk motion, for a zoomed second pass.

Body motion (walking, standing up) moves a large part of the action box at once;
hand work over the desk moves small regions. So per sampled frame pair we take the
motion inside the desk band of the crop box, drop pairs where much of the box moves,
smooth over time, and return the strongest peaks as (start, end, zoom_box).

Usage:
    python windows.py video.mov            # print windows
    python windows.py video.mov --debug out.png
"""
import argparse
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from crop import find_action_box


def _samples(path, box, fps):
    x0, y0, x1, y1 = box
    with av.open(path) as c:
        s = c.streams.video[0]
        src_fps = float(s.average_rate or 30)
        step = max(1, round(src_fps / fps))
        times, frames = [], []
        for i, frame in enumerate(c.decode(s)):
            if i % step:
                continue
            g = frame.to_image().convert("L").crop(box)
            g = g.resize((g.width // 2, g.height // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))
            frames.append(np.asarray(g, np.float32))
            times.append(i / src_fps)
    return np.array(times), frames


def find_windows(path, box=None, fps=8.0, desk_band=(0.45, 1.0), body_frac=0.12,
                 win=5.0, max_windows=3, zoom_pad=0.25):
    """Return (windows, debug). Each window: dict(start, end, peak, score, zoom_box)."""
    box = box or find_action_box(path)[0]
    times, frames = _samples(path, box, fps)
    h, w = frames[0].shape
    b0, b1 = int(desk_band[0] * h), int(desk_band[1] * h)

    energy = np.zeros(len(frames))
    heat = np.zeros((h, w), np.float32)
    for k in range(1, len(frames)):
        d = frames[k] - frames[k - 1]
        moving = np.abs(d - np.median(d)) > 12
        if moving.mean() > body_frac:          # whole body / lighting: not hand work
            continue
        band = moving[b0:b1]
        energy[k] = band.mean()
        heat[b0:b1] += band

    # Smooth over ~1 s and pick separated peaks.
    k = max(1, int(fps))
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    order = np.argsort(smooth)[::-1]
    picked = []
    for i in order:
        if smooth[i] <= 0 or len(picked) == max_windows:
            break
        if all(abs(times[i] - times[j]) > win for j in picked):
            picked.append(i)

    dur = times[-1]
    bx0, by0, bx1, by1 = box
    windows = []
    for i in sorted(picked):
        t = times[i]
        start, end = max(0.0, t - win / 2), min(dur, t + win / 2)
        # Zoom box around where the desk motion happened inside this window.
        sel = [j for j in range(1, len(frames)) if start <= times[j] <= end]
        local = np.zeros((h, w), np.float32)
        for j in sel:
            d = frames[j] - frames[j - 1]
            m = np.abs(d - np.median(d)) > 12
            if m.mean() <= body_frac:
                local[b0:b1] += m[b0:b1]
        ys, xs = np.nonzero(local > 0.3 * local.max()) if local.max() > 0 else ([], [])
        if len(xs):
            zx0, zx1, zy0, zy1 = xs.min(), xs.max(), ys.min(), ys.max()
            pw, ph = (zx1 - zx0) * zoom_pad + 8, (zy1 - zy0) * zoom_pad + 8
            zb = (max(0, zx0 - pw), max(0, zy0 - ph), min(w, zx1 + pw), min(h, zy1 + ph))
            zoom = tuple(int(v) for v in (bx0 + zb[0] * 2, by0 + zb[1] * 2, bx0 + zb[2] * 2, by0 + zb[3] * 2))
        else:
            zoom = box
        windows.append({"start": round(start, 1), "end": round(end, 1), "peak": round(t, 1),
                        "score": round(float(smooth[i]), 4), "zoom_box": zoom})
    return windows, {"times": times, "energy": smooth, "box": box}


def save_debug(path, windows, debug, out_png, width=900):
    t, e = debug["times"], debug["energy"]
    H = 160
    img = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(img)
    dur = max(t[-1], 1e-6)
    em = max(e.max(), 1e-6)
    for wdw in windows:
        d.rectangle([wdw["start"] / dur * width, 0, wdw["end"] / dur * width, H], fill=(255, 230, 180))
    pts = [(ti / dur * width, H - 10 - ei / em * (H - 30)) for ti, ei in zip(t, e)]
    d.line(pts, fill=(30, 90, 200), width=2)
    for s in range(0, int(dur) + 1, 5):
        d.text((s / dur * width + 2, H - 12), f"{s}s", fill="black")
    img.save(out_png)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--debug")
    args = ap.parse_args()
    windows, debug = find_windows(args.video)
    for wdw in windows:
        print(wdw)
    if args.debug:
        save_debug(args.video, windows, debug, args.debug)


if __name__ == "__main__":
    main()
