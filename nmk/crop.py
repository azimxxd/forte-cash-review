"""Find the rectangle of a video where the action happens and write a cropped copy.

Algorithm (motion heatmap, deliberately coarse so corner noise doesn't win):
  1. Sample frames at ~4 fps, grayscale, downscale to ~120 px wide, light blur.
  2. Diff consecutive samples. Subtract each diff's median so global lighting
     changes (lights on/off, auto-exposure) cancel out; skip pairs where most of
     the frame changed anyway (a cut or a big exposure jump).
  3. Accumulate thresholded diffs into a heatmap, blur it, and threshold it at a
     fraction of its peak.
  4. Keep connected blobs whose motion mass is at least `keep_ratio` of the
     biggest blob. Small blobs (a flickering screen in a corner, a passer-by
     behind glass) are dropped.
  5. Take the bounding box of the kept blobs, trim the outer 2% of motion mass
     on each side, pad by `pad` of the frame size, enforce a minimum size, and
     round to multiples of 32 px (the model's patch grid).

Usage:
    python crop.py video.mov                 # print the crop box, save a debug overlay
    python crop.py video.mov --write out.mp4 # also write the cropped video
"""
import argparse
import os
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _gray_samples(path, sample_fps=4.0, work_width=120):
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 30)
        step = max(1, round(fps / sample_fps))
        out, size = [], None
        for i, frame in enumerate(c.decode(s)):
            if i % step:
                continue
            img = frame.to_image()
            size = img.size
            h = round(img.height * work_width / img.width)
            g = img.convert("L").resize((work_width, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))
            out.append(np.asarray(g, dtype=np.float32))
    return out, size


def _components(mask):
    """4-connected components of a boolean mask -> list of (pixel_indices) arrays."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    comps, cur = [], 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if labels[y0, x0]:
            continue
        cur += 1
        stack, pix = [(y0, x0)], []
        labels[y0, x0] = cur
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = cur
                    stack.append((ny, nx))
        comps.append(np.array(pix))
    return comps


def find_action_box(path, pad=(0.15, 0.02, 0.15, 0.12), keep_ratio=0.2, heat_level=0.2,
                    min_frac=0.4, trim=0.03, align=32):
    """Return ((x0, y0, x1, y1), debug) in original-pixel coordinates.

    pad is a fraction of the frame size: one number, or (left, top, right, bottom)."""
    pl, pt, pr, pb = pad if isinstance(pad, (tuple, list)) else (pad,) * 4
    frames, (W, H) = _gray_samples(path)
    if len(frames) < 2:
        return (0, 0, W, H), None
    h, w = frames[0].shape

    heat = np.zeros((h, w), np.float32)
    used = 0
    for a, b in zip(frames, frames[1:]):
        d = b - a
        d = np.abs(d - np.median(d))  # cancels uniform brightness shifts
        moving = d > 12
        if moving.mean() > 0.5:      # whole frame changed: lighting/cut, not action
            continue
        heat += moving
        used += 1
    if used == 0 or heat.max() == 0:
        return (0, 0, W, H), heat

    smooth = np.asarray(Image.fromarray(heat / heat.max() * 255).convert("L")
                        .filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    mask = smooth > heat_level * smooth.max()

    comps = _components(mask)
    masses = [heat[c[:, 0], c[:, 1]].sum() for c in comps]
    biggest = max(masses)
    kept = np.zeros_like(mask)
    for c, m in zip(comps, masses):
        if m >= keep_ratio * biggest:
            kept[c[:, 0], c[:, 1]] = True

    # Bounding box of kept blobs, trimming the outermost motion mass.
    weight = heat * kept
    def span(profile):
        cdf = np.cumsum(profile) / profile.sum()
        return int(np.searchsorted(cdf, trim)), int(np.searchsorted(cdf, 1 - trim)) + 1
    gx0, gx1 = span(weight.sum(0))
    gy0, gy1 = span(weight.sum(1))

    sx, sy = W / w, H / h
    x0, x1, y0, y1 = gx0 * sx, gx1 * sx, gy0 * sy, gy1 * sy
    x0, x1 = x0 - pl * W, x1 + pr * W
    y0, y1 = y0 - pt * H, y1 + pb * H

    def fit(a0, a1, full):
        size = min(full, max(a1 - a0, min_frac * full))
        size = min(full, int(np.ceil(size / align)) * align) if full >= align else full
        c = (a0 + a1) / 2
        a0 = int(round(min(max(c - size / 2, 0), full - size)))
        return a0, a0 + size
    x0, x1 = fit(x0, x1, W)
    y0, y1 = fit(y0, y1, H)
    return (x0, y0, x1, y1), {"heat": heat, "kept": kept, "pairs_used": used, "pairs": len(frames) - 1}


def save_debug(path, box, debug, out_png):
    with av.open(path) as c:
        img = next(c.decode(video=0)).to_image().convert("RGB")
    if debug is not None:
        heat = debug["heat"] / max(debug["heat"].max(), 1e-6)
        hm = Image.fromarray((heat * 255).astype(np.uint8)).resize(img.size)
        red = Image.new("RGB", img.size, (255, 0, 0))
        img = Image.composite(red, img, hm.point(lambda v: int(v * 0.6)))
    d = ImageDraw.Draw(img)
    d.rectangle(box, outline=(0, 255, 0), width=3)
    img.save(out_png)


def write_cropped(path, box, out_path):
    x0, y0, x1, y1 = box
    with av.open(path) as src, av.open(out_path, "w") as dst:
        s = src.streams.video[0]
        o = dst.add_stream("libx264", rate=s.average_rate)
        o.width, o.height, o.pix_fmt = x1 - x0, y1 - y0, "yuv420p"
        o.options = {"crf": "18", "preset": "veryfast"}
        for frame in src.decode(s):
            arr = frame.to_ndarray(format="rgb24")[y0:y1, x0:x1]
            for p in o.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")):
                dst.mux(p)
        for p in o.encode():
            dst.mux(p)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--write", help="write the cropped video here")
    ap.add_argument("--debug", help="save a heatmap + box overlay PNG here")
    ap.add_argument("--pad", type=float, nargs="+", help="1 or 4 values (left top right bottom), fraction of frame")
    args = ap.parse_args()

    box, debug = find_action_box(args.video, **({"pad": args.pad[0] if len(args.pad) == 1 else tuple(args.pad)} if args.pad else {}))
    with av.open(args.video) as c:
        s = c.streams.video[0]
        W, H = s.codec_context.width, s.codec_context.height
    x0, y0, x1, y1 = box
    print(f"{os.path.basename(args.video)}: {W}x{H} -> crop x={x0}..{x1} y={y0}..{y1} "
          f"({x1 - x0}x{y1 - y0}, {100 * (x1 - x0) * (y1 - y0) / (W * H):.0f}% of area)"
          + (f", {debug['pairs_used']}/{debug['pairs']} frame pairs used" if debug else ""))
    if args.debug:
        save_debug(args.video, box, debug, args.debug)
    if args.write:
        write_cropped(args.video, box, args.write)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
