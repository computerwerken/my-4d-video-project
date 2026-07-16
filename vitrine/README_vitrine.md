# jg4d vitrine kit

Floats scanned archival material in a museum case — curled paper, scuffed glass,
dust in a light cone, black all around — aimed at red-cyan anaglyph delivery.

Drop `vitrine_kit.py` next to `jg4d_ldi3_player.py` in `blender/`, and
`vitrine_prep.py` + `vitrine_prep_open.jsx` in a new `tools/` (or wherever you
keep the prep side). They share the LDI3 player's conventions: same `jg4d`
N-panel tab, same JSON-sidecar idea, same numpy-only, no-Geometry-Nodes,
no-drivers house style. Nothing here depends on the LDI3 player, and vice versa —
but both are ordinary Cycles scenes, so a vitrine can sit inside a captured LDI3
scene later if the film wants that.

## The two stages

```
scan / PDF / web jpeg
        |
        |  vitrine_prep.py          headless, no Photoshop, batches a whole PDF
        v
  <stem>_albedo.png   16-bit sRGB   descreened, illumination-flattened colour
  <stem>_alpha.png    16-bit gray   the matte
  <stem>_ink.png      16-bit gray   ink coverage -> dots + bump + gloss
  <stem>_scuff.png    16-bit gray   glass wear (optional; the addon is procedural)
  <stem>.json                       analysis + real physical size
        |
        |  vitrine_kit.py           Blender addon: N-panel > jg4d > vitrine
        v
  Cycles multiview L/R  ->  your existing Resolve -> Dubois chain
```

```bash
# one page
python3 vitrine_prep.py scan.tif -o items --width-mm 127

# a whole book
python3 vitrine_prep.py The_Farm_Book.pdf --pages 12,20,44-52 -o items --dpi 300

# a folder of web downloads
python3 vitrine_prep.py ~/Downloads/archive/ -o items

# mix freely; PDFs, files and folders in one go
python3 vitrine_prep.py book.pdf --pages 20 poster.tif ~/Downloads/refs/ -o items
```

## Inputs

| Input | Support | Notes |
|---|---|---|
| **TIFF, 16-bit** | **best** | what to give it if you control the scan |
| PNG, 16-bit | best | same depth, bigger files |
| TIFF / PNG, 8-bit | good | fine; all the analysis still works |
| **PDF** (scanned) | **good** | `--pages 3,7,20-24`; pulls the embedded raster at native res, no re-render |
| PDF (designed: posters, flyers) | good | detected (largest embedded image ≪ page) and the page is rendered instead |
| JPEG | ok | 8-bit, and its noise sits near the screen frequency |
| WEBP, BMP, GIF, JP2… | ok | anything Pillow/OpenCV opens; 8-bit |
| AVIF, HEIC | ok | needs `pip3 install pillow-heif` |
| Transparent PNG/WebP | good | the file's own alpha becomes the matte; color under transparency is rebuilt for clean edges |
| **https:// URL** | good | fetched into the work folder, then treated as the above |
| Folder | — | every supported image inside, non-recursive |

**Web images have no real physical size.** 72/96 dpi metadata is a software
default, not a measurement, and is ignored. Give `--width-mm`, or a
`--preset` (`poster onesheet a2 a3 a4 letter tabloid postcard photo ticket
card`), or the item is assumed 300 ppi and flagged `size_assumed` in its
sidecar and QC. Size drives the stereo solve, dust scale and DOF — a poster
prepped at postcard scale will stereo-solve like a postcard.

**Low-coherence "screens" are left alone.** When the periodic peaks don't agree
on one frequency (coherence < 0.5 — repeated text lines, JPEG block hash),
auto mode now applies point-notches only instead of the wedge stop, which used
to smear real content on designed posters. Force the wedge with
`--descreen wedge` when you know it really is one screen.

**Bit depth is the axis that matters, not the container.** A 16-bit TIFF and a
16-bit PNG are identical here; an 8-bit TIFF and a JPEG are near enough the same
too. The prep does real tonal lifts (flat-field division, density normalisation)
and 8 bits banks in the shadows under that — banding that survives all the way to
the anaglyph, where it lands as per-eye noise. Web-sourced 8-bit JPEGs are fine,
because 8 bits is all that ever existed for them and there's nothing to recover.
But when you scan something yourself, scan 16-bit.

The tool prints the depth it actually got (`2551x3300 px  16-bit in`) and records
it in the sidecar, so a truncation can't pass unnoticed.

**Resolution:** aim for ~1.5× the pixels you need on screen. `--target-height`
(default 2160) makes the tool tell you: *downscaling 1.53x - good* vs
*UPSCALING 0.24x - soft*. For an upscale, run it through UpscaleVideo.ai from
this repo first, then prep the result.

**Scale:** `--width-mm` is worth setting for anything not letter-size, since it
drives the real-world size in Blender, which drives dust scale, DOF and the
stereo solve. Otherwise ppi comes from `--dpi`, then the file's own metadata,
then a 300 assumption.

**PDFs** use `pdfimages` (poppler) to pull the embedded image bit-exact, or
PyMuPDF to render if there's no raster. Without either, extract the pages
yourself and feed the images.

Then in Blender: **N > jg4d > vitrine**, pick any `*_albedo.png`, **Build
vitrine**, **Solve stereo**. The siblings and the sidecar are found for you.

Needs numpy + Pillow. Uses scipy / opencv / PyMuPDF if they happen to be there.

## What it decided about your Farm Book scan

`The_Farm_Book.pdf` — 108 pages, 2551×3300 at 300 ppi, 8-bit, moderate JPEG,
Canon capture on a copy stand.

| | measured | verdict |
|---|---|---|
| resolution | 2551×3300 | fine — **1.53× downscale** at 2160 px tall |
| halftone | **133 lpi @ 46°, 2.26 px/cell** | **88% of Nyquist — already aliasing in the scan** |
| clipping | 0.00% white, 0.03% black | clean, nothing to recover |
| illumination | ~16–23% falloff | corrected |
| paper | RGB 0.80/0.76/0.67 | warm; **kept** (aged tone is content, not error) |

Usable. The resolution is genuinely fine. The halftone is the whole problem, and
it's why the pipeline is shaped the way it is.

## The three things this gets right that are easy to get wrong

**1. The halftone is already aliased, so don't rebuild it faithfully.**
133 lpi sampled at 300 ppi is 2.26 px per cell — 88% of Nyquist. The beat you
can see in the scan *is* the aliasing; it's baked in. The prep notches the
lattice out (85% suppressed) and exports clean ink *density*. The addon then
rebuilds a halftone procedurally, and Cycles anti-aliases it properly with its
pixel filter — sampling done right this time.

But rebuilding the *real* 133 lpi is a trap: a 279 mm sheet filling a 2160 px
frame gives **1.48 px per cell**, so you'd recreate exactly the aliasing you just
removed. **Safe ruling** computes what the frame can carry (~3 px/cell → **65
lpi**) and defaults to it. The dots then read as ink — catching the raking light,
lifting the gloss — instead of as sampling error. Run it again if you change lens,
distance or resolution.

**2. 63 mm interaxial is wrong here, by about 7×.**
That's the LDI3 player's default and it's correct for a captured room. A vitrine
is a small subject half a metre away. For a 260 mm case at 420 mm on a 70 mm lens
at 4096×2160:

```
63 mm  ->  +7.15% parallax at the case back   (unfusable)
solved ->  8.8 mm for a 1% budget             ("Solve stereo" gives you this)
```

**Solve stereo** computes it from the actual geometry and your parallax target
rather than assuming a face.

**3. Convergence goes on the glass.**
Then the theatre screen *is* the vitrine pane: the item sits behind it in positive
parallax, the scuffs land at zero parallax where they're sharpest, and dust drifts
up toward the glass but never past it. No window violation is possible by
construction — verified, not hoped: the dust drift amplitude is a hard bound, so
motes stay behind the pane (~22 mm clearance at the defaults).

## Randomize

Every random thing comes from one integer seed, so a look is reproducible and
each button is just *seed + 1, rebuild that part*:

- **Curl** — bend axis, bow, corner lift, twist, ripple. Reshapes in place.
- **Glass** — new wipe arcs, scratches, smudges. Procedural, so it's instant and
  file-free. `Use painted scuff map` overrides with your `*_scuff.png`.
- **Dust** — motes redistributed through the case.
- **Lighting** — bounded by **Light variation** (default 0.25, deliberately low:
  you asked for *slightly*). Position, energy and Kelvin jitter around your values.

## Notes for delivery

- **Solve stereo** sets Blender's built-in anaglyph for *preview only* — it's a
  plain matrix, not Dubois. For delivery, set `views_format` to **Individual** and
  put the L/R masters through your grade, then the existing `jg4d.dubois`
  operator (or your Resolve chain). Same as everything else in the film.
- **Volumetric noise is per-eye noise.** Left and right denoise independently, and
  mismatched noise is the one thing red-cyan will not fuse. Give the volume real
  samples rather than leaning on the denoiser.
- **Keep motes dim.** Bright sub-pixel specks are ghosting bait in anaglyph.
- The paper is lit, not emissive — deliberately. An emissive item sits outside the
  volume's lighting and reads as a sticker on the black rather than a thing in a
  case.

## Known limits

- **Deep vignettes.** Past ~25% falloff, shaded paper in the corners can be darker
  than lit ink in the middle, and no single-image estimate fully separates them.
  The tool corrects what it's confident about, says so, and tells you to check.
  Measured: 25% → flat, 45% → flat, 60% → about half corrected. Your scan is ~16–23%.
- **Genuinely periodic subjects** (textiles, brickwork, engraved guilloche) look
  like screens to the detector. Use `--no-descreen` for those.
- **Pillow cannot be trusted with 16 bits in either direction**, so this file
  goes around it on both ends. Writing: Pillow has no 16-bit RGB mode and
  `frombytes('RGB;16B')` silently truncates, so the PNG encoder here is ~15 lines
  of hand-rolled zlib (verified via ImageMagick and an independent decode). Note
  Pillow *reads* those files back as 8-bit — a Pillow limitation, not the file;
  Blender and Photoshop are fine. Reading: a 16-bit RGB TIFF opens with mode
  already `'RGB'` (halved), and a 16-bit *gray* TIFF opens as `'I;16'` where
  `.convert('RGB')` assumes a 0–255 range and turns a 512-level ramp into **three
  levels**. Neither raises. `load_rgb()` uses OpenCV's `IMREAD_UNCHANGED` first
  and falls back carefully.

## Tuning quick reference

| Symptom | Knob |
|---|---|
| screen survives the descreen | `--wedge-deg 45`, `--drift 0.08` |
| descreen too soft | `--descreen wedge` (skips the roll-off) |
| item looks flattened *into* the page | `--flatten 0.5` |
| dots invisible | raise **Dot relief**, not **Dot in colour** |
| dots shimmer under a move | **Safe ruling**, then go lower still |
| case reads as a doll's house | camera back, or shorten the case, then **Solve stereo** |
| glass invisible | raise **Sheen W**, or give it a **Room HDRI** to reflect |
