#!/usr/bin/env python3
# jg4d vitrine prep
# -----------------
# Turns a scanned/downloaded archival image into a four-part material kit for
# the jg4d vitrine kit Blender addon. Headless: no Photoshop required, batches
# a whole PDF in one go. Ships a companion .jsx if you want the result back as
# a layered PSD for hand work (see vitrine_prep_open.jsx).
#
# Per item it writes:
#   <stem>_albedo.png   16-bit sRGB  - descreened, illumination-flattened color
#   <stem>_alpha.png    16-bit gray  - the matte (paper silhouette, real edges)
#   <stem>_ink.png      16-bit gray  - ink coverage/density (drives bump+gloss+
#                                      the procedural halftone in Blender)
#   <stem>_scuff.png    16-bit gray  - glass wear for the vitrine pane (seeded)
#   <stem>.json                      - analysis + physical size, read by the addon
#
# Why "ink" and not "paper grain": a 300 ppi capture cannot resolve paper fibre
# (tooth is ~20-50 um; you'd need 1200+ ppi). What a high-pass of a 300 ppi scan
# actually contains is the PRINTING SCREEN, aliased. Feeding that to a bump map
# gives you a vibrating lattice, which in a red-cyan anaglyph is ghosting bait.
# So: we notch the screen out here, export clean ink DENSITY, and the addon
# rebuilds a controllable halftone procedurally at render resolution. Paper
# tooth is synthesised in the shader, where it can be scaled to the real sheet.
#
# Deps: numpy + Pillow (required). scipy / opencv / PyMuPDF used if present.
#
# MIT License.

import argparse
import json
import math
import os
import re
import subprocess
import sys
import urllib.parse

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None

# HEIC/AVIF: increasingly what "save image" gives you from the web. Optional.
try:
    import pillow_heif as _heif
    _heif.register_heif_opener()
    try:
        _heif.register_avif_opener()
    except Exception:
        pass
except Exception:
    _heif = None

# Everything the pipeline will take as a single image. PDFs are handled
# separately. One list, imported by the app, so the two never drift.
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".tif", ".tiff", ".webp",
                  ".bmp", ".dib", ".gif", ".jp2", ".avif", ".heic", ".heif")

# Real-world sizes for things that arrive as bare pixels (a downloaded poster
# has no trustworthy ppi). Width in mm; height follows the pixel aspect.
SIZE_PRESETS = {
    "poster": 610.0,        # 24x36 in
    "onesheet": 686.0,      # 27x41 in movie one-sheet
    "a2": 420.0, "a3": 297.0, "a4": 210.0,
    "letter": 216.0, "tabloid": 279.0,
    "postcard": 152.0, "photo": 102.0,   # 4x6 in
    "card": 63.5, "ticket": 82.0,
}


# ----------------------------------------------------------------------------
# Small image utilities (numpy fallbacks so the tool runs anywhere)
# ----------------------------------------------------------------------------

def blur(a, sigma):
    """Gaussian-ish blur. scipy if available, else 3x box (CLT ~ gaussian)."""
    if sigma <= 0:
        return a.copy()
    if _ndi is not None:
        if a.ndim == 3:
            return np.stack([_ndi.gaussian_filter(a[..., c], sigma, mode="nearest")
                             for c in range(a.shape[2])], axis=2)
        return _ndi.gaussian_filter(a, sigma, mode="nearest")
    # box-blur fallback: 3 passes of width w approximates gaussian of this sigma
    w = max(1, int(round(sigma * 1.7724 / 3) * 2 + 1))
    out = a.astype(np.float64)
    for _ in range(3):
        out = _box(out, w)
    return out


def _box(a, w):
    r = w // 2
    if r < 1:
        return a
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r, r)
        p = np.pad(a, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        zero = np.zeros_like(np.take(c, [0], axis=axis))
        c = np.concatenate([zero, c], axis=axis)
        n = a.shape[axis]
        hi = np.take(c, np.arange(w, w + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        a = (hi - lo) / float(w)
    return a


def maxfilt(a, size):
    """True square maximum filter with edge replication.

    The fallback is separable and exact -- an earlier version strided the
    offsets and used np.roll, which produced a sparse comb of shifted copies
    that also wrapped around the array. Silent, and wrong everywhere it mattered.
    """
    size = max(1, int(size) | 1)
    if size == 1:
        return a.copy()
    if _ndi is not None:
        return _ndi.maximum_filter(a, size=size, mode="nearest")
    if _cv2 is not None and a.dtype in (np.float32, np.float64, np.uint8):
        # Do NOT cast to float32 here. Callers compare `x == maxfilt(x)` to find
        # local maxima; a float64 -> float32 -> float64 round trip perturbs the
        # low bits and that equality then silently never holds, so peak detection
        # quietly returns nothing. cv2.dilate handles CV_64F natively.
        return _cv2.dilate(np.ascontiguousarray(a),
                           np.ones((size, size), np.uint8),
                           borderType=_cv2.BORDER_REPLICATE)
    r = size // 2
    out = a
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        stack = [np.take(p, np.arange(k, k + a.shape[axis]), axis=axis)
                 for k in range(size)]
        out = np.maximum.reduce(stack)
    return out


def load_rgb(path):
    rgb, _a, depth = load_rgba(path)
    return rgb, depth


def load_rgba(path):
    """Any image -> (float64 RGB in [0,1], alpha in [0,1] or None, bit depth).

    Web "clipouts" (transparent PNG/WebP posters, scans someone already cut
    out) carry their matte in the file. That alpha is exact and hand-made;
    guessing a new one from luminance would be strictly worse -- and the RGB
    under transparent pixels is undefined (often black), which used to read
    as "51% black clipping" and a lucky dark-matte guess.

    Pillow alone cannot read depth properly and fails in two silent ways:
      * a 16-bit RGB TIFF opens with mode already 'RGB', i.e. 8-bit. Halved.
      * a 16-bit GRAY TIFF opens as 'I;16', and .convert('RGB') assumes a 0-255
        range, so a 512-level ramp comes out with THREE levels. Destroyed.
    Neither raises. Both matter here: a 16-bit flatbed scan is exactly the input
    this pipeline claims to want, and the tonal lifts downstream are where thin
    bit depth turns into banding you cannot grade out.

    So: OpenCV's IMREAD_UNCHANGED first (the only one of the three that reads
    16-bit properly), numpy-direct for Pillow's integer modes, 8-bit convert last.
    """
    if _cv2 is not None:
        try:
            a = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
        except Exception:
            a = None
        if a is not None and a.size:
            src_alpha = None
            if a.ndim == 2:
                a = np.repeat(a[:, :, None], 3, axis=2)
            else:
                if a.shape[2] == 4:
                    src_alpha = a[:, :, 3].astype(np.float64)
                    src_alpha /= 65535.0 if a.dtype == np.uint16 else \
                        (255.0 if a.dtype == np.uint8 else 1.0)
                    a = a[:, :, :3]
                if a.shape[2] == 3:
                    a = a[:, :, ::-1]              # cv2 is BGR
                elif a.shape[2] == 1:
                    a = np.repeat(a, 3, axis=2)
            if a.dtype == np.uint16:
                return np.ascontiguousarray(a).astype(np.float64) / 65535.0, src_alpha, 16
            if a.dtype == np.uint8:
                return np.ascontiguousarray(a).astype(np.float64) / 255.0, src_alpha, 8
            if a.dtype in (np.float32, np.float64):
                return np.clip(np.ascontiguousarray(a).astype(np.float64), 0, 1), src_alpha, 32

    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        a = np.asarray(im).astype(np.float64)
        scale = 65535.0 if a.max() > 255 else 255.0
        return np.repeat((a / scale)[:, :, None], 3, axis=2), None, 16
    if im.mode == "F":
        a = np.clip(np.asarray(im).astype(np.float64), 0, 1)
        return np.repeat(a[:, :, None], 3, axis=2), None, 32

    src_alpha = None
    if "A" in im.getbands() or "transparency" in im.info:
        rgba = np.asarray(im.convert("RGBA")).astype(np.float64) / 255.0
        src_alpha = rgba[..., 3]
    out = np.asarray(im.convert("RGB")).astype(np.float64) / 255.0
    # Pillow hands back a 16-bit RGB file already flattened to 8. Ask the
    # container what it actually claims to be, so the loss is reported as OUR
    # missing dependency rather than looking like the scanner's fault.
    declared = _declared_depth(path)
    if declared and declared > 8:
        print("   ! %s declares %d bits per channel but Pillow can only give 8 "
              "for RGB.\n     Install OpenCV to read it at full depth: "
              "pip install opencv-python-headless" % (os.path.basename(path), declared))
    return out, src_alpha, 8


def _declared_depth(path):
    """What the file's own header claims, without trusting the decoder."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".png":
            with open(path, "rb") as f:
                head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                return int(head[24])
        elif ext in (".tif", ".tiff"):
            with Image.open(path) as im:
                bps = getattr(im, "tag_v2", {}).get(258)
                if bps is not None:
                    return int(max(bps)) if hasattr(bps, "__iter__") else int(bps)
    except Exception:
        pass
    return None


def luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


def _png_chunk(tag, data):
    import struct
    import zlib
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _save16_impl(path, arr):
    """Write a true 16-bit PNG (gray or RGB).

    Pillow has no 16-bit RGB mode -- 'RGB' is 8 bits per channel, and
    frombytes('RGB;16B') silently truncates. Since the whole point of this
    stage is a 16-bit handoff (these are 1.5-stop lifts on aged paper; 8 bits
    banks in the shadows and that banding survives all the way to the anaglyph),
    we emit the PNG ourselves. It is ~15 lines and removes a dependency.
    """
    import struct
    import zlib
    arr = np.ascontiguousarray(arr.astype(">u2"))
    if arr.ndim == 2:
        h, w = arr.shape
        color_type = 0
    else:
        h, w, nc = arr.shape
        color_type = {1: 0, 3: 2, 4: 6}[nc]
    raw = arr.tobytes()
    stride = len(raw) // h
    # filter byte 0 (None) per scanline
    lines = b"".join(b"\x00" + raw[i * stride:(i + 1) * stride] for i in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", zlib.compress(lines, 6)))
        f.write(_png_chunk(b"IEND", b""))


def save16(path, a):
    """a: HxW or HxWx3 float in [0,1] -> 16-bit PNG."""
    _save16_impl(path, (np.clip(a, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16))


# ----------------------------------------------------------------------------
# Periodic-screen detection
#
# A halftone screen is a coherent lattice: in an averaged tile spectrum it is a
# handful of sharp isolated spikes, while pictorial content decorrelates across
# tiles and averages into a smooth 1/f background. So: average |FFT| over a
# grid of windowed tiles, subtract a radial background, and any spike that
# survives is periodic. This finds the fundamental, its harmonics, AND the
# aliases those harmonics fold back to -- which is what the visible moire beat
# actually is, and what a naive "notch the fundamental" descreen leaves behind.
# ----------------------------------------------------------------------------

TILE = 512


def _tile_spectrum(lum, tile=TILE, max_tiles=48):
    h, w = lum.shape
    tile = min(tile, h, w)
    tile -= tile % 2
    if tile < 64:
        return None, tile
    ys = list(range(0, h - tile + 1, max(tile // 2, 1)))
    xs = list(range(0, w - tile + 1, max(tile // 2, 1)))
    coords = [(y, x) for y in ys for x in xs]
    if not coords:
        coords = [(0, 0)]
    if len(coords) > max_tiles:
        step = len(coords) / float(max_tiles)
        coords = [coords[int(i * step)] for i in range(max_tiles)]
    win = np.outer(np.hanning(tile), np.hanning(tile))
    acc = np.zeros((tile, tile), np.float64)
    for (y, x) in coords:
        t = lum[y:y + tile, x:x + tile].astype(np.float64)
        t = t - t.mean()
        acc += np.abs(np.fft.fftshift(np.fft.fft2(t * win)))
    return acc / len(coords), tile


def _radial_background(mag, tile):
    """Median magnitude per radius: the smooth 1/f floor to measure spikes against."""
    c = tile // 2
    yy, xx = np.mgrid[0:tile, 0:tile]
    rr = np.hypot(yy - c, xx - c)
    ri = rr.astype(np.int32)
    nb = ri.max() + 1
    med = np.zeros(nb)
    mad = np.zeros(nb)
    flat = mag.ravel()
    idx = ri.ravel()
    order = np.argsort(idx)
    idx_s, flat_s = idx[order], flat[order]
    bounds = np.searchsorted(idx_s, np.arange(nb + 1))
    for k in range(nb):
        seg = flat_s[bounds[k]:bounds[k + 1]]
        if seg.size:
            m = np.median(seg)
            med[k] = m
            mad[k] = np.median(np.abs(seg - m)) + 1e-12
    return med[ri], mad[ri], rr


def detect_screen(lum, min_freq=0.02, snr=9.0, max_peaks=24):
    """Return (peaks, meta). peaks: list of dicts with fx,fy (cycles/px, signed
    from DC), ruling in cycles/px, angle in degrees, strength in MAD units."""
    mag, tile = _tile_spectrum(lum)
    if mag is None:
        return [], {"note": "image too small for screen analysis"}
    med, mad, rr = _radial_background(mag, tile)
    score = (mag - med) / (mad * 1.4826)

    c = tile // 2
    fr = rr / float(tile)                       # cycles/px
    score[fr < min_freq] = 0.0
    score[fr > 0.495] = 0.0

    # local maxima in a 5x5 neighbourhood. Compared with a tolerance rather than
    # ==: whichever backend maxfilt uses, exact float equality is a bad hinge to
    # hang the whole detector on.
    mx = maxfilt(score, 5)
    peak_mask = (score >= mx - 1e-9 * np.maximum(np.abs(mx), 1.0)) & (score > snr)

    ys, xs = np.nonzero(peak_mask)
    cand = sorted(zip(score[ys, xs], ys, xs), reverse=True)[:max_peaks]

    peaks = []
    for s, y, x in cand:
        fy = (y - c) / float(tile)
        fx = (x - c) / float(tile)
        f = math.hypot(fx, fy)
        peaks.append({
            "fx": float(fx), "fy": float(fy),
            "freq_cyc_px": float(f),
            "period_px": float(1.0 / f) if f > 0 else 0.0,
            "angle_deg": float(math.degrees(math.atan2(fy, fx)) % 180.0),
            "strength": float(s),
        })
    return peaks, {"tile": tile}


def screen_metric(rgb, r0, tile=TILE):
    """Mean spectral magnitude in the lattice annulus, divided by a quiet
    reference band. Scale-free, so before/after numbers are comparable and a
    value near 0 genuinely means the screen is gone (not just 'peak got wider')."""
    if not r0:
        return 0.0
    lum = luma(rgb)
    mag, tile = _tile_spectrum(lum, tile=tile)
    if mag is None:
        return 0.0
    c = tile // 2
    yy, xx = np.mgrid[0:tile, 0:tile]
    fr = np.hypot(yy - c, xx - c) / float(tile)
    lat = (fr > r0 - 0.03) & (fr < r0 + 0.03)
    ref = (fr > 0.14) & (fr < 0.24)
    if lat.sum() == 0 or ref.sum() == 0:
        return 0.0
    return float(mag[lat].mean() / max(mag[ref].mean(), 1e-9))


def summarize_screen(peaks, ppi):
    """Describe the lattice in print terms, using the SAME r0 the descreen
    decision uses. Reporting one number and acting on another is how you end up
    reading '12% of Nyquist' next to a mode that only triggers above 30%."""
    if not peaks:
        return None
    r0, angles, coherence = _screen_axes(peaks)
    if not r0:
        return None
    p = max(peaks, key=lambda q: q["strength"])
    return {
        "ruling_lpi": round(r0 * ppi, 1) if ppi else None,
        "angle_deg": round(angles[0], 1) if angles else round(p["angle_deg"], 1),
        "axes_deg": [round(a, 1) for a in angles],
        "period_px": round(1.0 / r0, 2) if r0 else 0.0,
        "pct_of_nyquist": round(r0 / 0.5 * 100.0, 1),
        "coherence": round(coherence, 2),
        "strength": round(p["strength"], 1),
        "n_periodic_peaks": len(peaks),
    }


# ----------------------------------------------------------------------------
# Descreen: overlap-add notch filtering
#
# Tiled because a camera capture of a bound book has perspective and page curl,
# so the screen frequency drifts across the sheet -- a single global notch would
# only catch the middle. Each tile re-detects locally, seeded by the global fit.
# ----------------------------------------------------------------------------

def _screen_axes(peaks, min_strength=0.25, tol=0.15):
    """Reduce the peak list to the lattice's radius and its axis angles.

    r0 is the radius of the STRONGEST peak -- that is what a fundamental is.
    An earlier version took the median radius of all strong peaks, which is
    right only when they already agree. On a clean scan they do (the Farm Book's
    sit tightly at 0.44), so it looked fine. On anything where the peaks scatter
    -- a downsampled scan whose screen has been aliased into hash -- the median
    lands at a frequency where no actual peak lives, and the wedge then goes and
    notches empty spectrum while the real peaks sail through.

    Angles are only those peaks sharing r0 (within `tol`), because that is what
    a second lattice axis IS. Peaks at other radii are harmonics or junk; they
    still get individually notched, they just do not get to define the wedge.
    """
    if not peaks:
        return None, [], 0.0
    strongest = max(peaks, key=lambda p: p["strength"])
    r0 = float(strongest["freq_cyc_px"])
    smax = strongest["strength"]
    strong = [p for p in peaks if p["strength"] >= smax * min_strength]

    at_r0 = [p for p in strong if abs(p["freq_cyc_px"] - r0) <= tol * r0]
    angles = []
    for p in sorted(at_r0, key=lambda q: -q["strength"]):
        a = p["angle_deg"]
        if not any(min(abs(a - b), 180 - abs(a - b)) < 12 for b in angles):
            angles.append(a)

    # How much of the strong energy actually agrees on r0. Near 1 = a real
    # lattice. Low = scattered periodic hash, and the caller should say so
    # rather than quietly pretending it found a screen.
    tot = sum(p["strength"] for p in strong) or 1.0
    coherence = sum(p["strength"] for p in at_r0) / tot
    return r0, angles, float(coherence)


def build_mask(tile, peaks, mode="auto", sigma_bins=3.0, wedge_deg=30.0,
               drift=0.05, lp_rolloff=None, axes=None):
    """Frequency-domain suppression mask for one tile.

    Three cooperating parts, because one alone does not do it:
      * point notches on the detected lattice spikes -- exact, cheap, and the
        only correct tool for a WELL-sampled screen (fine scan of coarse print),
        where real detail lives on both sides of the screen frequency.
      * an angular WEDGE x radial band stop. A camera capture of a bound book
        has perspective and curl, so the screen's frequency and angle drift
        across the sheet: the spike is really a smear, and point notches only
        catch its middle. The wedge follows the whole smear while leaving the
        other ~300 degrees of that frequency band untouched.
      * an optional gentle roll-off ABOVE the screen. Only engaged when the
        screen is near Nyquist, where the capture's own MTF guarantees there is
        nothing real left to protect -- just screen and JPEG mosquito noise.
    """
    c = tile // 2
    yy, xx = np.mgrid[0:tile, 0:tile].astype(np.float64)
    fr = np.hypot(yy - c, xx - c) / float(tile)
    m = np.ones((tile, tile), np.float64)

    for p in peaks:
        py, px = c + p["fy"] * tile, c + p["fx"] * tile
        for (qy, qx) in ((py, px), (2 * c - py, 2 * c - px)):   # Hermitian pair
            m *= 1.0 - np.exp(-((yy - qy) ** 2 + (xx - qx) ** 2)
                              / (2.0 * sigma_bins ** 2))

    # The lattice STRUCTURE (how many axes, at what angles) is a property of the
    # printed page, not of this 512px tile, so it is passed in from the global
    # detection. Only the frequency drifts across a curled sheet, and that is
    # what the per-tile refinement tracks. Re-deriving the axis set per tile
    # loses a whole lattice wherever one happens to be locally weaker -- on this
    # page (two screens, 46/137 and 16/106) that cost 11 points of suppression.
    r0, angles, _coh = axes if axes else _screen_axes(peaks)
    if mode in ("wedge", "wedge+lp", "auto") and r0 and angles:
        ang = np.degrees(np.arctan2(yy - c, xx - c)) % 180.0
        # The radial width must scale with r0 and must never reach DC. A fixed
        # 0.05 cyc/px band is harmless around a near-Nyquist screen at 0.44, but
        # around a well-sampled screen at 0.03 it swallows the origin -- and
        # notching DC removes the image's mean, i.e. the whole page goes dark.
        sig_r = min(drift, r0 * 0.30)
        band = np.exp(-((fr - r0) ** 2) / (2.0 * sig_r ** 2))
        band *= (fr > r0 * 0.5)          # hard floor: DC and its neighbourhood are sacred
        for a0 in angles:
            da = np.minimum(np.abs(ang - a0), 180.0 - np.abs(ang - a0))
            m *= 1.0 - np.exp(-(da ** 2) / (2.0 * (wedge_deg / 2.0) ** 2)) * band

    if lp_rolloff:
        m *= 1.0 / (1.0 + (fr / lp_rolloff) ** 8)      # Butterworth-ish, no ringing

    m[c, c] = 1.0                        # never touch DC, whatever else happened
    return m


def descreen(rgb, peaks, tile=TILE, sigma_bins=3.0, mode="auto",
             wedge_deg=30.0, drift=0.05, lp_rolloff=None, local_refine=True,
             min_freq=0.02):
    """Suppress the periodic lattice in every channel. Returns (out, residual).

    Weighted overlap-add (analysis window, filter, synthesis window, normalise
    by the accumulated window energy) so the tiling is exactly transparent where
    the mask is 1 -- no seams, no gain error.
    """
    if not peaks:
        return rgb.copy(), np.zeros(rgb.shape[:2], np.float64)

    h, w = rgb.shape[:2]
    tile = min(tile, h, w)
    tile -= tile % 2
    if tile < 64:
        return rgb.copy(), np.zeros((h, w), np.float64)
    hop = tile // 2
    win = np.outer(np.hanning(tile), np.hanning(tile))

    # Pad on ALL FOUR sides, not just right/bottom. A Hann window is zero at its
    # own edges, so the first and last half-tile of the array are covered only by
    # the vanishing tail of one window: wsum -> 0 there and the normalisation
    # divides ~0 by ~0. Padding by half a tile pushes that dead zone into padding
    # we then crop off, so every surviving pixel sees full overlap.
    pad = hop
    ey = (-(h + 2 * pad - tile)) % hop
    ex = (-(w + 2 * pad - tile)) % hop
    src = np.pad(rgb, ((pad, pad + ey), (pad, pad + ex), (0, 0)), mode="reflect")
    H, W = src.shape[:2]
    acc = np.zeros((H, W, 3), np.float64)
    wsum = np.zeros((H, W), np.float64)

    kw = dict(mode=mode, sigma_bins=sigma_bins, wedge_deg=wedge_deg,
              drift=drift, lp_rolloff=lp_rolloff, axes=_screen_axes(peaks))
    global_mask = build_mask(tile, peaks, **kw)
    c = tile // 2
    boxes = [(c + p["fy"] * tile, c + p["fx"] * tile) for p in peaks]

    for y in range(0, H - tile + 1, hop):
        for x in range(0, W - tile + 1, hop):
            block = src[y:y + tile, x:x + tile, :]
            mask = global_mask
            if local_refine:
                t = luma(block)
                F0 = np.fft.fftshift(np.fft.fft2((t - t.mean()) * win))
                lp = _refine_peaks(np.abs(F0), boxes, tile, min_freq)
                if lp:
                    mask = build_mask(tile, lp, **kw)
            for ch in range(3):
                F = np.fft.fftshift(np.fft.fft2(block[..., ch] * win))
                filt = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
                acc[y:y + tile, x:x + tile, ch] += filt * win   # synthesis window
            wsum[y:y + tile, x:x + tile] += win * win

    cover = wsum[pad:pad + h, pad:pad + w]
    if cover.min() < 1e-3:
        print("   ! descreen: window coverage dropped to %.2e; widen the pad" % cover.min())
    out = (acc / np.maximum(wsum, 1e-9)[..., None])[pad:pad + h, pad:pad + w, :]
    resid = luma(rgb) - luma(out)
    return np.clip(out, 0.0, 1.0), resid


def _refine_peaks(mag, boxes, tile, min_freq, r=6):
    """Snap each global peak to the local maximum within +-r bins in this tile.
    Keeps the strength so _screen_axes can still tell trunk from tail."""
    c = tile // 2
    out = []
    for (py, px) in boxes:
        y0, y1 = int(max(0, py - r)), int(min(tile, py + r + 1))
        x0, x1 = int(max(0, px - r)), int(min(tile, px + r + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        sub = mag[y0:y1, x0:x1]
        k = int(np.argmax(sub))
        dy, dx = divmod(k, sub.shape[1])
        fy = (y0 + dy - c) / float(tile)
        fx = (x0 + dx - c) / float(tile)
        f = math.hypot(fx, fy)
        if f < min_freq:
            continue
        out.append({"fy": fy, "fx": fx, "freq_cyc_px": f,
                    "angle_deg": math.degrees(math.atan2(fy, fx)) % 180.0,
                    "strength": float(sub.flat[k])})
    return out


# ----------------------------------------------------------------------------
# Illumination flattening
#
# Divides out the LIGHTING gradient of the capture while PRESERVING the paper's
# intrinsic tone. That distinction matters for an archival documentary: the
# sepia of aged newsprint is content, the 28% falloff of somebody's Canon on a
# copy stand is not. Use --neutralize only if you want the paper white balanced.
# ----------------------------------------------------------------------------

def flatten_illumination(rgb, alpha=None, strength=1.0, order=2, iters=4):
    """Robust low-order flat-field.

    The field is fitted as a 2D POLYNOMIAL (quadratic by default), not taken
    from a local maximum. That choice is the whole point. A local-max field
    follows whatever is under it, so on a page with a big dark photograph it
    dips inside the photo and the "flatten" then brightens the photo by ~30% --
    it eats the picture and calls it lighting. A copy-stand's falloff is
    inverse-square times cos^4: smooth, monotone, and captured almost exactly by
    a quadratic. A quadratic has six degrees of freedom for the whole sheet, so
    it physically cannot chase a subject.

    Fitted iteratively against the bright (paper) samples, dropping whatever
    sits below the current fit -- ink is one-sided noise, so plain least squares
    would be dragged down by it.
    """
    h, w = rgb.shape[:2]
    lum = luma(rgb)

    # coarse grid of paper-white candidates: high percentile per cell
    gy, gx = 48, max(8, int(48 * w / max(h, 1)))
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    pts, vals, texs, cell_ij = [], [], [], []
    for a in range(gy):
        for b in range(gx):
            cell = lum[ys[a]:ys[a + 1], xs[b]:xs[b + 1]]
            if cell.size < 4:
                continue
            if alpha is not None:
                ac = alpha[ys[a]:ys[a + 1], xs[b]:xs[b + 1]]
                if ac.mean() < 0.5:
                    continue
            pts.append(((ys[a] + ys[a + 1]) * 0.5 / h - 0.5,
                        (xs[b] + xs[b + 1]) * 0.5 / w - 0.5))
            vals.append(np.percentile(cell, 90))
            # spread within the cell: bare paper is smooth, anything printed on
            # it is not. This is the half of the question brightness cannot answer.
            texs.append(np.percentile(cell, 90) - np.percentile(cell, 10))
            cell_ij.append((a, b))
    if len(pts) < 12:
        return rgb.copy(), 0.0
    P = np.array(pts)
    V = np.array(vals)
    T = np.array(texs)

    def design(yy, xx):
        cols = []
        for p in range(order + 1):
            for q in range(order + 1 - p):
                cols.append((yy ** p) * (xx ** q))
        return np.stack(cols, axis=-1)

    A = design(P[:, 0], P[:, 1])

    # Which cells are PAPER? A cell is paper if it is bright RELATIVE TO ITS
    # NEIGHBOURHOOD -- not relative to the whole sheet.
    #
    # Every global test fails somewhere. Plain (even robust) least squares gets
    # dragged down by ink: a big dark photograph in the middle of a page IS shaped
    # like a vignette, so a quadratic fits it happily and then "corrects" the
    # picture out of existence. Asymmetric least squares overshoots the other way,
    # bowing above 1.0 at the corners. A global Otsu works until the vignette is
    # deep enough that shaded paper in the corners is darker than lit ink in the
    # middle -- then the classes invert and it flattens the wrong thing.
    #
    # Local is the honest question, because the two effects differ in SCALE, not
    # in brightness: illumination is smooth over the whole sheet, ink is not. Ask
    # each cell whether it is near the brightest thing within a third of a frame
    # of it, and a vignette cannot fool you -- it barely varies over that span.
    Vg = np.full((gy, gx), np.nan)
    for (a, b), v in zip(cell_ij, V):
        Vg[a, b] = v
    filled = np.where(np.isnan(Vg), np.nanmedian(Vg), Vg)
    ref = maxfilt(filled, max(3, (gy // 3) | 1))
    bright_g = filled >= 0.85 * ref
    bright = np.array([bright_g[a, b] for (a, b) in cell_ij], bool)
    smooth = T <= max(np.percentile(T, 45), 0.02)
    paper = bright & smooth
    for relax in (None, "bright", "quantile"):
        if paper.sum() >= max(12, A.shape[1] * 3):
            break
        paper = bright if relax == "bright" else V >= np.percentile(V, 65)
    coef = None
    for _ in range(max(iters, 3)):
        if paper.sum() < A.shape[1] * 2:
            break
        coef, *_ = np.linalg.lstsq(A[paper], V[paper], rcond=None)
        r = V - A @ coef
        s = np.median(np.abs(r[paper] - np.median(r[paper]))) * 1.4826 + 1e-6
        new = paper & (r > -2.5 * s) & (r < 3.0 * s)
        if new.sum() < A.shape[1] * 2 or np.array_equal(new, paper):
            break
        paper = new
    if coef is None:
        return rgb.copy(), 0.0

    yy, xx = np.mgrid[0:h, 0:w]
    field = design(yy / h - 0.5, xx / w - 0.5) @ coef
    # The field is a lighting estimate, not a licence. Clamp it to the range the
    # paper samples actually support so an extrapolating quadratic cannot invent
    # a 1.5x corner out past the last real sample.
    field = np.clip(field, V[paper].min() * 0.9, V[paper].max() * 1.05)
    field = np.maximum(field, 1e-3)

    ref = float(np.median(field))
    gain = 1.0 + (ref / field - 1.0) * float(strength)
    gain = np.clip(gain, 0.4, 2.5)
    out = np.clip(rgb * gain[..., None], 0.0, 1.0)
    falloff = float(1.0 - field.min() / max(field.max(), 1e-6))
    return out, falloff


def neutralize(rgb, alpha=None):
    """White-balance the paper to neutral. Off by default -- the tone is content."""
    lum = luma(rgb)
    m = lum > np.percentile(lum[alpha > 0.5] if alpha is not None else lum, 90)
    if alpha is not None:
        m &= alpha > 0.5
    if m.sum() < 64:
        return rgb
    ref = np.array([rgb[..., c][m].mean() for c in range(3)])
    g = ref.mean() / np.maximum(ref, 1e-4)
    return np.clip(rgb * g[None, None, :], 0.0, 1.0)


def fill_under_alpha(rgb, alpha, steps=24):
    """Extend edge colors into fully-transparent regions (masked dilation at
    reduced size, then a soft merge). Not archival content -- just keeps
    undefined RGB out of the analysis and out of texture-filter taps."""
    h, w = rgb.shape[:2]
    scale = max(1, int(max(h, w) / 512))
    small = rgb[::scale, ::scale].copy()
    a = (alpha[::scale, ::scale] > 0.5).astype(np.float64)
    filled = small * a[..., None]
    weight = a.copy()
    for _ in range(steps):
        if weight.min() > 0:
            break
        fb = _box(filled, 3)
        wb = _box(weight[..., None], 3)[..., 0]
        newly = (weight < 0.5) & (wb > 1e-4)
        filled[newly] = (fb[newly] / np.maximum(wb[newly, None], 1e-6))
        weight[newly] = 1.0
    filled = np.where(weight[..., None] > 0, filled, small)
    if scale > 1:
        big = np.stack([np.asarray(Image.fromarray(
            (np.clip(filled[..., c], 0, 1) * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR)) for c in range(3)], axis=2) / 255.0
    else:
        big = filled
    keep = np.clip(alpha, 0.0, 1.0)[..., None]
    return rgb * keep + big * (1.0 - keep)


# ----------------------------------------------------------------------------
# Matte
# ----------------------------------------------------------------------------

def make_matte(rgb, mode="auto", feather=1.2):
    h, w = rgb.shape[:2]
    lum = luma(rgb)

    if mode == "auto":
        mode = _guess_matte_mode(rgb, lum)

    if mode == "full":
        a = np.ones((h, w), np.float64)
        return a, mode

    if mode == "white":
        sat = rgb.max(2) - rgb.min(2)
        a = 1.0 - np.clip((lum - 0.86) / 0.10, 0, 1) * (1.0 - np.clip(sat / 0.06, 0, 1))
    elif mode == "dark":
        a = np.clip((lum - 0.04) / 0.10, 0, 1)
    else:  # page: the sheet is the bright connected region on a darker ground
        thr = _otsu(lum)
        a = (lum > thr).astype(np.float64)
        a = _largest_blob(a)

    a = np.clip(a, 0, 1)
    if feather > 0:
        a = blur(a, feather)
    return np.clip(a, 0, 1), mode


def _guess_matte_mode(rgb, lum):
    """Look at the border ring: bright+neutral -> white knockout; dark -> dark
    knockout; neither (the frame is full of paper) -> no cut, it's a snippet."""
    h, w = lum.shape
    k = max(2, int(min(h, w) * 0.02))
    ring = np.concatenate([lum[:k].ravel(), lum[-k:].ravel(),
                           lum[:, :k].ravel(), lum[:, -k:].ravel()])
    sat = rgb.max(2) - rgb.min(2)
    ringsat = np.concatenate([sat[:k].ravel(), sat[-k:].ravel(),
                              sat[:, :k].ravel(), sat[:, -k:].ravel()])
    if ring.mean() > 0.90 and ringsat.mean() < 0.05 and ring.std() < 0.08:
        return "white"
    if ring.mean() < 0.10 and ring.std() < 0.08:
        return "dark"
    return "full"


def _otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins, range=(0, 1))
    hist = hist.astype(np.float64)
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(bins))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b))
    return edges[k]


def _largest_blob(mask):
    if _ndi is not None:
        lab, n = _ndi.label(mask > 0.5)
        if n == 0:
            return mask
        sizes = _ndi.sum(mask > 0.5, lab, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        out = (lab == keep).astype(np.float64)
        return _ndi.binary_fill_holes(out > 0.5).astype(np.float64)
    if _cv2 is not None:
        m8 = (mask > 0.5).astype(np.uint8)
        n, lab, stats, _ = _cv2.connectedComponentsWithStats(m8, 8)
        if n <= 1:
            return mask
        keep = 1 + int(np.argmax(stats[1:, _cv2.CC_STAT_AREA]))
        return (lab == keep).astype(np.float64)
    return mask


# ----------------------------------------------------------------------------
# Ink density map
#
# NOT a high-pass of the raw scan (that is the aliased screen -- see header).
# This is band-limited ink COVERAGE from the descreened image: how much ink sits
# on the sheet at this point, 0 = bare paper, 1 = solid. In Blender it drives
# three things at once: a micron-scale bump, a gloss delta (ink is smoother than
# paper), and the threshold for the resynthesised halftone.
# ----------------------------------------------------------------------------

def make_ink_map(rgb_desc, alpha=None, ppi=300.0, highpass_mm=6.0):
    lum = luma(rgb_desc)
    if alpha is not None and (alpha > 0.5).sum() > 64:
        paper = np.percentile(lum[alpha > 0.5], 97)
        solid = np.percentile(lum[alpha > 0.5], 0.5)
    else:
        paper = np.percentile(lum, 97)
        solid = np.percentile(lum, 0.5)
    density = np.clip((paper - lum) / max(paper - solid, 1e-3), 0.0, 1.0)

    # Remove tonal MASSES so a dark photograph is not one big lump of relief;
    # keep the marks. sigma = highpass_mm of real paper.
    sigma_px = max(2.0, highpass_mm / 25.4 * ppi)
    band = density - blur(density, sigma_px)
    band = 0.5 + band * 0.5 / max(np.percentile(np.abs(band), 99.5), 1e-3) * 0.5
    ink = np.clip(density * 0.75 + (band - 0.5) * 0.5 + 0.0, 0.0, 1.0)
    if alpha is not None:
        ink *= np.clip(alpha, 0, 1)
    return ink, {"paper_level": float(paper), "solid_level": float(solid)}


# ----------------------------------------------------------------------------
# Glass scuff map
#
# White-on-black, the way you'd document a real case: long shallow wipe arcs
# from a decade of cloth in the same handful of directions, a few hard isolated
# scratches, one or two faint smudge patches. Seeded, so the addon's "randomize
# glass" can also just ask for a different seed. The addon ALSO has a fully
# procedural node version; this file is the override slot for hand-painting.
# ----------------------------------------------------------------------------

def make_scuff_map(size=2048, seed=0, wipes=14, scratches=7, smudges=3,
                   intensity=1.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float64)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    # wipe arcs: cloth pivots from a shoulder, so arcs share a few big centres
    n_pivots = int(rng.integers(2, 4))
    pivots = [(rng.uniform(-1.5, 2.5) * size, rng.uniform(-1.5, 2.5) * size)
              for _ in range(n_pivots)]
    for _ in range(wipes):
        cy, cx = pivots[int(rng.integers(0, n_pivots))]
        R = math.hypot(size * 0.5 - cx, size * 0.5 - cy) * rng.uniform(0.7, 1.3)
        width = rng.uniform(1.5, 7.0)
        d = np.abs(np.hypot(yy - cy, xx - cx) - R)
        arc = np.exp(-(d ** 2) / (2 * width ** 2))
        # only part of the arc is present, and it fades along its length
        ang = np.arctan2(yy - cy, xx - cx)
        a0 = rng.uniform(-math.pi, math.pi)
        span = rng.uniform(0.25, 1.1)
        da = np.abs(((ang - a0 + math.pi) % (2 * math.pi)) - math.pi)
        seg = np.clip(1.0 - da / span, 0, 1) ** 0.7
        img += arc * seg * rng.uniform(0.10, 0.34)

    # hard scratches: thin, straight, high contrast, no fade
    for _ in range(scratches):
        y0, x0 = rng.uniform(0, size, 2)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(0.05, 0.42) * size
        y1, x1 = y0 + math.sin(ang) * L, x0 + math.cos(ang) * L
        d = _seg_dist(yy, xx, y0, x0, y1, x1)
        img += np.exp(-(d ** 2) / (2 * rng.uniform(0.5, 1.3) ** 2)) * rng.uniform(0.5, 1.0)

    # smudges: low-frequency, soft, barely there
    lowf = _value_noise(size, rng, cells=6)
    for _ in range(smudges):
        cy, cx = rng.uniform(0.15, 0.85, 2) * size
        r = rng.uniform(0.06, 0.19) * size
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        img += blob * lowf * rng.uniform(0.06, 0.15)

    # dust film / micro-haze so the pane is never mathematically clean
    img += _value_noise(size, rng, cells=size // 4) * 0.035

    img *= intensity
    return np.clip(img, 0.0, 1.0)


def _seg_dist(yy, xx, y0, x0, y1, x1):
    vy, vx = y1 - y0, x1 - x0
    L2 = vy * vy + vx * vx + 1e-9
    t = np.clip(((yy - y0) * vy + (xx - x0) * vx) / L2, 0, 1)
    return np.hypot(yy - (y0 + t * vy), xx - (x0 + t * vx))


def _value_noise(size, rng, cells=8):
    g = rng.random((cells + 1, cells + 1))
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    a = np.asarray(im).astype(np.float64) / 255.0
    return (a - a.min()) / max(a.max() - a.min(), 1e-6)


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------

def parse_pages(spec):
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def pdf_pages(path, pages, workdir, dpi_hint=None):
    """Extract embedded page images at native resolution. Prefers pdfimages
    (bit-exact: no resampling, no re-render), falls back to PyMuPDF.

    The bit-exact path assumes the page IS one big scan (a scanned book). A
    designed PDF -- a poster, a flyer, anything vector -- has small embedded
    images (logos, photos) and the "largest image on the page" is then some
    logo, not the page. So: compare the best embedded image against the page's
    own area; if it covers well under half the page at 150 dpi, render the
    page instead."""
    got = []
    have_pdfimages = _which("pdfimages")
    n = _pdf_page_count(path)
    page_pts = _pdf_page_size_pts(path)
    todo = pages or list(range(1, n + 1))
    for p in todo:
        stem = os.path.join(workdir, "_pdfpage_%04d" % p)
        if have_pdfimages:
            try:
                subprocess.run(["pdfimages", "-j", "-p", "-f", str(p), "-l", str(p),
                                path, stem], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cand = sorted(f for f in os.listdir(workdir)
                              if f.startswith(os.path.basename(stem)))
                if cand:
                    # largest extracted image on the page = the page scan?
                    best = max(cand, key=lambda f: os.path.getsize(os.path.join(workdir, f)))
                    bestp = os.path.join(workdir, best)
                    ok = True
                    if page_pts:
                        try:
                            with Image.open(bestp) as bi:
                                bpx = bi.size[0] * bi.size[1]
                            expect = (page_pts[0] / 72.0 * 150) * (page_pts[1] / 72.0 * 150)
                            ok = bpx >= 0.35 * expect
                        except Exception:
                            ok = True
                    if ok:
                        got.append((p, bestp))
                        continue
                    print("  page %d: largest embedded image covers <35%% of the page "
                          "(designed PDF, not a scan) -- rendering the page instead" % p)
            except Exception:
                pass
        try:
            import fitz
            doc = fitz.open(path)
            pg = doc[p - 1]
            pix = pg.get_pixmap(dpi=int(dpi_hint or 300))
            out = stem + ".png"
            pix.save(out)
            got.append((p, out))
            doc.close()
        except Exception as e:
            print("  ! page %d: cannot extract (%s)" % (p, e))
    return got


def _pdf_page_size_pts(path):
    try:
        out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
        m = re.search(r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", out)
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception:
        pass
    try:
        import fitz
        d = fitz.open(path)
        r = d[0].rect
        d.close()
        return float(r.width), float(r.height)
    except Exception:
        return None


def _pdf_page_count(path):
    try:
        out = subprocess.run(["pdfinfo", path], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    try:
        import fitz
        d = fitz.open(path)
        n = d.page_count
        d.close()
        return n
    except Exception:
        return 0


def _which(prog):
    from shutil import which
    return which(prog) is not None


def guess_ppi(img_path, w_px, args):
    """Returns (ppi, assumed). `assumed` is True when nothing trustworthy said
    what the physical size is -- which is the normal case for anything saved
    off the web. 72 and 96 dpi are ignored on purpose: they are the software
    DEFAULTS (Mac/Web and Windows respectively) that image editors stamp on
    files that were never scanned, not measurements. Trusting them turned an
    arbitrary 900 px download into a 318 mm "sheet"."""
    if args.dpi:
        return float(args.dpi), False
    wmm = args.width_mm or SIZE_PRESETS.get(getattr(args, "preset", None) or "")
    if wmm:
        return w_px / (wmm / 25.4), False
    try:
        with Image.open(img_path) as im:
            d = im.info.get("dpi")
            if d and d[0] and 20 < d[0] < 4800 and \
               not (71 <= d[0] <= 73) and not (95 <= d[0] <= 97):
                return float(d[0]), False
    except Exception:
        pass
    return 300.0, True


def fetch_url(url, workdir):
    """Download an online document into the work folder and return its path.
    Extension comes from the URL, falling back to Content-Type."""
    import urllib.request
    os.makedirs(workdir, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                  os.path.basename(urllib.parse.urlparse(url).path)) or "download"
    req = urllib.request.Request(url, headers={"User-Agent":
        "Mozilla/5.0 (jg4d-vitrine-prep; archival tooling)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTS + (".pdf",):
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
               "image/tiff": ".tif", "image/gif": ".gif", "image/bmp": ".bmp",
               "image/avif": ".avif", "image/heic": ".heic", "image/heif": ".heif",
               "application/pdf": ".pdf"}.get(ctype, "")
        if not ext:
            raise RuntimeError("cannot tell what %s is (Content-Type %r); "
                               "save it locally and prep the file" % (url, ctype))
        name = os.path.splitext(name)[0] + ext
    out = os.path.join(workdir, name)
    with open(out, "wb") as f:
        f.write(data)
    print("   fetched %s (%.1f MB, %s)" % (name, len(data) / 1048576.0, ctype or "?"))
    return out


# ----------------------------------------------------------------------------
# Per-item pipeline
# ----------------------------------------------------------------------------

def stem_of(img_path, out_dir, taken=None):
    """Filesystem-safe stem, deduplicated: poster.bmp and poster.gif in one
    batch must not silently overwrite each other's kits."""
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                  os.path.splitext(os.path.basename(img_path))[0])
    stem, k = base, 2
    while (taken is not None and stem in taken) or \
          (taken is None and False):
        stem = "%s_%d" % (base, k)
        k += 1
    if taken is not None:
        taken.add(stem)
    return stem


def process(img_path, out_dir, args, stem=None, page=None, taken=None):
    stem = stem or stem_of(img_path, out_dir, taken)
    print("\n== %s" % stem)

    rgb, src_alpha, src_depth = load_rgba(img_path)
    h, w = rgb.shape[:2]

    ppi, size_assumed = guess_ppi(img_path, w, args)
    width_mm = args.width_mm or SIZE_PRESETS.get(getattr(args, "preset", None) or "") \
        or (w / ppi * 25.4)
    height_mm = width_mm * h / w
    print("   %dx%d px  %d-bit in  @ ~%.0f ppi  -> %.0f x %.0f mm sheet%s"
          % (w, h, src_depth, ppi, width_mm, height_mm,
             "  (ASSUMED)" if size_assumed else ""))
    if size_assumed:
        print("     ! no trustworthy physical size in this file (web image?). "
              "Assumed 300 ppi.\n       Set --width-mm or --preset "
              "(%s) -- size drives the stereo solve,\n       dust scale and DOF."
              % ", ".join(sorted(SIZE_PRESETS)))

    if src_alpha is not None and float(src_alpha.min()) > 0.98:
        src_alpha = None                      # fully opaque: not a real matte
    if src_alpha is not None:
        # The file brought its own matte. Use it, and rebuild the color under
        # transparent pixels from the nearest opaque neighbours so descreen /
        # ink analysis and Blender's bilinear taps never see the undefined
        # (usually black) RGB that lives there.
        rgb = fill_under_alpha(rgb, src_alpha)
    if src_depth <= 8:
        levels = len(np.unique((luma(rgb) * 255).astype(np.uint8)))
        if levels < 200:
            print("     note: only %d distinct levels; thin for the tonal lifts "
                  "downstream. Fine for web-sourced items, but scan your own at 16-bit." % levels)

    lum = luma(rgb)

    # --- analysis ---
    peaks, meta = detect_screen(lum, snr=args.screen_snr)
    scr = summarize_screen(peaks, ppi)
    if scr:
        print("   screen: %.0f lpi @ %s deg, %.2f px pitch, %.0f%% of Nyquist "
              "(%d peaks, coherence %.2f)"
              % (scr["ruling_lpi"], "/".join("%.0f" % a for a in scr["axes_deg"]) or "?",
                 scr["period_px"], scr["pct_of_nyquist"], scr["n_periodic_peaks"],
                 scr["coherence"]))
        if scr["pct_of_nyquist"] > 70:
            print("     ^ undersampled: the beat you see IS aliasing. Descreen strongly advised.")
        r0n = scr["period_px"]
        if 7.5 <= r0n <= 8.5 and all(min(a % 90, 90 - a % 90) < 4 for a in scr["axes_deg"]):
            print("     ^ 8 px pitch on the pixel axes = JPEG block grid, not a printing "
                  "screen.\n       Notching it is harmless (mild deblocking).")
    else:
        print("   screen: none detected (continuous tone, or already descreened)")

    clip_hi = float((lum > 0.985).mean() * 100)
    clip_lo = float((lum < 0.015).mean() * 100)
    print("   clipping: %.2f%% white, %.2f%% black" % (clip_hi, clip_lo))

    # effective resolution against the delivery frame
    eff = None
    if args.target_height:
        eff = h / float(args.target_height)
        verdict = ("downscaling %.2fx - good" % eff if eff >= 1.15 else
                   "near 1:1 - fine" if eff >= 0.95 else
                   "UPSCALING %.2fx - soft; consider UpscaleVideo.ai from this repo" % eff)
        print("   at %d px tall on screen: %s" % (args.target_height, verdict))

    # --- descreen ---
    ds_mode, lp = args.descreen, None
    if peaks and not args.no_descreen:
        r0, angles, coh = _screen_axes(peaks)
        if ds_mode == "auto":
            # A screen above ~0.30 cyc/px is undersampled by the capture itself.
            # Above it there is no recoverable picture -- the lens never passed
            # it -- so we may also roll off, which no well-sampled scan should.
            if r0 and r0 > 0.30:
                ds_mode = "wedge+lp"
                lp = min(0.46, r0 * 0.95)
            else:
                ds_mode = "wedge"
        elif ds_mode == "wedge+lp":
            lp = min(0.46, (r0 or 0.44) * 0.95)
        if coh < 0.5:
            print("     low lattice coherence (%.2f): the strong peaks disagree on a "
                  "frequency,\n       so this is scattered periodic content rather than "
                  "one screen." % coh)
            if args.descreen == "auto":
                # The wedge stop is justified BY a coherent lattice: it clears a
                # whole annulus sector on the theory that one screen drifts
                # through it. Without coherence that theory is false, and on web
                # material (posters full of repeated text lines, JPEG block
                # hash) the wedge smears real content. Point notches only.
                ds_mode, lp = "notch", None
                print("       -> auto: point-notches only (no wedge). Force it with "
                      "--descreen wedge;\n          skip entirely with --no-descreen "
                      "if the subject is genuinely patterned.")
        before = screen_metric(rgb, r0)
        rgb_d, resid = descreen(rgb, peaks, sigma_bins=args.notch_sigma,
                                mode=ds_mode, wedge_deg=args.wedge_deg,
                                drift=args.drift, lp_rolloff=lp,
                                local_refine=not args.no_refine)
        after = screen_metric(rgb_d, r0)
        print("   descreen [%s%s]: lattice energy %.3f -> %.3f (%.0f%% suppressed)" % (
            ds_mode, ", rolloff %.2f cyc/px" % lp if lp else "",
            before, after, 100.0 * (1 - after / max(before, 1e-9))))
    else:
        rgb_d, resid = rgb.copy(), np.zeros((h, w))
        ds_mode = "none"

    # --- matte ---
    if src_alpha is not None and args.matte == "auto":
        alpha, mode = np.clip(src_alpha, 0.0, 1.0), "source"
    else:
        alpha, mode = make_matte(rgb_d, args.matte, feather=args.feather)
    print("   matte: %s (%.1f%% coverage)" % (mode, alpha.mean() * 100))

    # --- illumination ---
    rgb_f, falloff = flatten_illumination(rgb_d, alpha, strength=args.flatten)
    print("   illumination falloff %.0f%% -> flattened at %.0f%% strength"
          % (falloff * 100, args.flatten * 100))
    if falloff > 0.25 and args.flatten > 0:
        # Past roughly a stop of falloff, shaded paper in the corners can be
        # darker than lit ink in the middle, and no single-image estimate can
        # fully separate the two. The tool corrects what it is confident about
        # and says so, rather than guessing and quietly damaging the item.
        print("     ! deep falloff: paper/ink separation gets ambiguous past ~25%.")
        print("       Expect a partial correction. Check the albedo; if the item "
              "looks flattened INTO the page, re-run with --flatten 0.5")
    if args.neutralize:
        rgb_f = neutralize(rgb_f, alpha)
        print("   paper white neutralized")

    paper_rgb = [float(np.average(rgb_f[..., c],
                 weights=np.maximum(alpha * (lum > np.percentile(lum, 90)), 1e-6)))
                 for c in range(3)]

    # --- ink ---
    ink, inkmeta = make_ink_map(rgb_f, alpha, ppi=ppi, highpass_mm=args.ink_highpass_mm)
    print("   ink coverage: mean %.3f, 99th %.3f" % (ink.mean(), np.percentile(ink, 99)))

    # --- scuff ---
    scuff = make_scuff_map(size=args.scuff_size, seed=args.seed,
                           intensity=args.scuff_intensity)

    # --- write ---
    os.makedirs(out_dir, exist_ok=True)
    p_alb = os.path.join(out_dir, stem + "_albedo.png")
    p_alp = os.path.join(out_dir, stem + "_alpha.png")
    p_ink = os.path.join(out_dir, stem + "_ink.png")
    p_scf = os.path.join(out_dir, stem + "_scuff.png")
    p_jsn = os.path.join(out_dir, stem + ".json")

    _save16_impl(p_alb, (np.clip(rgb_f, 0, 1) * 65535 + 0.5).astype(np.uint16))
    _save16_impl(p_alp, (np.clip(alpha, 0, 1) * 65535 + 0.5).astype(np.uint16))
    _save16_impl(p_ink, (np.clip(ink, 0, 1) * 65535 + 0.5).astype(np.uint16))
    _save16_impl(p_scf, (np.clip(scuff, 0, 1) * 65535 + 0.5).astype(np.uint16))

    sidecar = {
        "_comment": "jg4d vitrine item. Written by vitrine_prep.py; read by vitrine_kit.py.",
        "source": os.path.basename(img_path),
        "page": page,
        "pixels": [w, h],
        "source_bit_depth": src_depth,
        "ppi": round(ppi, 1),
        "physical_mm": [round(width_mm, 2), round(height_mm, 2)],
        "matte_mode": mode,
        "size_assumed": bool(size_assumed),
        "paper_rgb": [round(v, 4) for v in paper_rgb],
        "descreened": ds_mode,
        "screen": scr,
        "ink": {k: round(v, 4) for k, v in inkmeta.items()},
        "illumination_falloff": round(falloff, 3),
        "clipping_pct": {"white": round(clip_hi, 3), "black": round(clip_lo, 3)},
        "effective_scale_at_target": round(eff, 3) if eff else None,
        "scuff_seed": args.seed,
        # what the addon should rebuild, in real print terms
        "resynth_halftone": {
            "ruling_lpi": (scr["ruling_lpi"] if scr else 100.0),
            "angle_deg": (scr["angle_deg"] if scr else 45.0),
        },
        "maps": {
            "albedo": os.path.basename(p_alb),
            "alpha": os.path.basename(p_alp),
            "ink": os.path.basename(p_ink),
            "scuff": os.path.basename(p_scf),
        },
    }
    with open(p_jsn, "w") as f:
        json.dump(sidecar, f, indent=2)
    print("   wrote %s_{albedo,alpha,ink,scuff}.png + .json" % stem)
    return p_alb


# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="jg4d vitrine prep: scan -> 4-part material kit for Blender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="images, PDFs, directories, or http(s) URLs")
    ap.add_argument("-o", "--out", default="vitrine_items", help="output directory")
    ap.add_argument("--pages", help="PDF pages, e.g. 3,7,20-24")

    ap.add_argument("--dpi", type=float, help="override scan ppi")
    ap.add_argument("--width-mm", type=float,
                    help="real width of the object in mm (sets scale AND ppi)")
    ap.add_argument("--preset", choices=sorted(SIZE_PRESETS),
                    help="physical size preset for web images with no real ppi "
                         "(poster, a4, postcard, ...)")
    ap.add_argument("--target-height", type=int, default=2160,
                    help="px the item will occupy on the delivery frame (for the "
                         "resolution verdict); 0 to skip")

    ap.add_argument("--matte", default="auto",
                    choices=["auto", "page", "full", "white", "dark"])
    ap.add_argument("--feather", type=float, default=1.2, help="matte feather px")

    ap.add_argument("--no-descreen", action="store_true")
    ap.add_argument("--descreen", default="auto",
                    choices=["auto", "notch", "wedge", "wedge+lp"],
                    help="auto picks wedge+lp for undersampled screens (>0.30 "
                         "cyc/px), wedge otherwise")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip per-tile peak refinement (faster, worse on curled pages)")
    ap.add_argument("--screen-snr", type=float, default=9.0,
                    help="how many sigma above the spectral floor counts as a screen")
    ap.add_argument("--notch-sigma", type=float, default=3.0,
                    help="point-notch width in FFT bins")
    ap.add_argument("--wedge-deg", type=float, default=30.0,
                    help="angular width of the lattice wedge stop; raise if the "
                         "screen survives on a badly curled page")
    ap.add_argument("--drift", type=float, default=0.05,
                    help="radial width of the wedge stop in cyc/px (how much the "
                         "screen frequency wanders across the sheet)")

    ap.add_argument("--flatten", type=float, default=1.0,
                    help="illumination flatten strength, 0..1")
    ap.add_argument("--neutralize", action="store_true",
                    help="white-balance the paper (off: aged tone is content)")
    ap.add_argument("--ink-highpass-mm", type=float, default=6.0)

    ap.add_argument("--seed", type=int, default=0, help="glass scuff seed")
    ap.add_argument("--scuff-size", type=int, default=2048)
    ap.add_argument("--scuff-intensity", type=float, default=1.0)

    args = ap.parse_args(argv)
    if args.target_height == 0:
        args.target_height = None

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_work")
    os.makedirs(work, exist_ok=True)

    jobs = []
    for inp in args.inputs:
        if re.match(r"^https?://", inp, re.I):
            try:
                inp = fetch_url(inp, work)
            except Exception as e:
                print("  ! could not fetch %s: %s" % (inp, e))
                continue
        if os.path.isdir(inp):
            for f in sorted(os.listdir(inp)):
                if f.lower().endswith(SUPPORTED_EXTS):
                    jobs.append((os.path.join(inp, f), None, None))
        elif inp.lower().endswith(".pdf"):
            base = os.path.splitext(os.path.basename(inp))[0]
            base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
            for p, path in pdf_pages(inp, parse_pages(args.pages), work, args.dpi):
                jobs.append((path, "%s_p%03d" % (base, p), p))
        else:
            jobs.append((inp, None, None))

    if not jobs:
        print("nothing to do")
        return 1

    print("jg4d vitrine prep: %d item(s) -> %s" % (len(jobs), args.out))
    taken = set()
    for path, stem, page in jobs:
        try:
            process(path, args.out, args, stem=stem, page=page, taken=taken)
        except Exception as e:
            import traceback
            print("  ! failed on %s: %s" % (path, e))
            traceback.print_exc()
    print("\ndone. In Blender: sidebar (N) > jg4d > Vitrine > pick any *_albedo.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
