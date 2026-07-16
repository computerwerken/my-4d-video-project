# jg4d vitrine kit — setup, start to first frame

Written for macOS (you're on a Mac) and Blender 3.0+. Work through it once with a
single page from The Farm Book; the batch comes after it works.

Total: about 20 minutes, most of it Blender rendering.

---

## 0. Prerequisites

```bash
python3 --version        # need 3.8+; macOS ships 3.9, that's fine
```

```bash
python3 -m pip install --upgrade numpy pillow

# strongly recommended, not strictly required:
python3 -m pip install opencv-python-headless   # 16-bit reading + faster filters
python3 -m pip install scipy                    # faster still

# for PDF input:
brew install poppler
```

**What each one actually does for you:**

| | without it |
|---|---|
| numpy, Pillow | won't run at all |
| **opencv** | **a 16-bit RGB TIFF gets silently halved to 8-bit.** Pillow has no 16-bit RGB path. The tool now detects and warns, but it can't fix it. 16-bit *grayscale* is fine either way. |
| scipy | ~2× slower. No behaviour change. |
| poppler | PDF extraction falls back to PyMuPDF (`pip3 install pymupdf`) if present, else no PDF input |
| pillow-heif | no AVIF/HEIC input (common for web-saved posters) — `pip3 install pillow-heif` |

Check what you got:

```bash
python3 -c "import numpy,PIL;print('core ok')"
python3 -c "import cv2;print('cv2',cv2.__version__)" || echo "no cv2 (16-bit RGB will halve)"
which pdfimages || echo "no poppler (no PDF input)"
```

---

## 1. Put the files in your fork

```bash
cd ~/path/to/my-4d-video-project

cp vitrine_kit.py blender/                 # next to jg4d_ldi3_player.py
mkdir -p tools
cp vitrine_prep.py vitrine_prep_open.jsx tools/
cp README_vitrine.md blender/README_vitrine.md
```

`vitrine_kit.py` doesn't import the LDI3 player and the LDI3 player doesn't import
it. They share the `jg4d` N-panel tab and nothing else. Installing one does not
affect the other.

---

## 2. Prep one page

```bash
cd ~/path/to/my-4d-video-project

python3 tools/vitrine_prep.py ~/path/to/The_Farm_Book.pdf \
    --pages 20 \
    -o ~/vitrine_items \
    --dpi 300 \
    --target-height 2160
```

Takes ~20 s/page. You should see:

```
== The_Farm_Book_p020
   2551x3300 px  8-bit in  @ ~300 ppi  -> 216 x 279 mm sheet
   screen: 133 lpi @ 46 deg, 2.26 px pitch, 88% of Nyquist (24 periodic peaks)
     ^ undersampled: the beat you see IS aliasing. Descreen strongly advised.
   clipping: 0.00% white, 0.03% black
   at 2160 px tall on screen: downscaling 1.53x - good
   descreen [wedge+lp, rolloff 0.42 cyc/px]: lattice energy 0.602 -> 0.091 (85% suppressed)
   matte: full (100.0% coverage)
   illumination falloff 23% -> flattened at 100% strength
   ink coverage: mean 0.291, 99th 0.740
   wrote The_Farm_Book_p020_{albedo,alpha,ink,scuff}.png + .json
```

**Read those lines — they're the QC pass, not decoration.**

| line | what you want | if not |
|---|---|---|
| `8-bit in` | matches your source | if it says 8 and you scanned 16 → install opencv |
| `% of Nyquist` | any value; >70% means descreen matters | — |
| `descreen … suppressed` | **>70%** | raise `--wedge-deg 45 --drift 0.08` |
| `clipping` | both <1% | blown scan; nothing to recover |
| `downscaling ≥1.15x` | good | `UPSCALING` → UpscaleVideo.ai first |
| `illumination falloff` | <25% | >25% prints its own warning; try `--flatten 0.5` |

**Look at the albedo before going further.** Open
`~/vitrine_items/The_Farm_Book_p020_albedo.png` and zoom to 100% on the
photograph. The dot lattice should be gone and the text still crisp. That's the
one judgement call the tool can't make for you.

---

## 3. Install the addon

Blender → **Edit > Preferences > Add-ons**

- **Blender 3.0–4.1:** `Install…` button, pick `blender/vitrine_kit.py`
- **Blender 4.2+:** the **⌄** dropdown, top right → `Install from Disk…`

Then tick **jg4d vitrine kit** to enable it.

Press **N** in the 3D viewport → a **jg4d** tab appears with two panels
(`jg4d LDI3 player` and `jg4d vitrine kit`).

### Do this now, not later: run Blender from the Terminal

```bash
/Applications/Blender.app/Contents/MacOS/Blender
```

`Solve stereo` and `Safe ruling` print their real working — interaxial, hfov,
parallax at each plane, px-per-cell — to **stdout**. The N-panel only gets a
one-line summary, and on macOS there's no *Window > Toggle System Console*
(that's Windows-only). Launched from Terminal, you see everything. If a shader
socket name mismatches on your Blender version, this is also where it says so.

---

## 4. Set render resolution FIRST

**Output Properties (printer icon) → Resolution X/Y.**

DCI 4K: `4096 × 2160`. DCI 2K: `2048 × 1080`.

Do this **before** you hit Build. `Safe ruling` reads `resolution_y` to work out
how many pixels the sheet lands on, and picks the halftone ruling from that. Build
at 1920×1080 and then switch to 4K and your dots are ~2× too coarse.

---

## 5. Build

In **N > jg4d > jg4d vitrine kit**:

1. Click the **file field** → pick `~/vitrine_items/The_Farm_Book_p020_albedo.png`
   (just the albedo — alpha, ink, scuff and the .json are found automatically)
2. **Build vitrine**

You get: the curled sheet at its real 216×279 mm, a pedestal and back in near-black
velvet, a scuffed glass pane, a haze volume, 400 dust motes, three lights, a black
world, and a camera. `Auto safe ruling` is on, so the ruling is already set.

Viewport still looks black? That's correct — the world is black to camera by
design. Switch the viewport to **Rendered** (the rightmost sphere, top right) and
look **through the camera** (**Numpad 0**).

---

## 6. Frame the shot — then re-solve

Move the camera however you like. Then, **in this order**:

1. **Safe ruling**
2. **Solve stereo**

**Both depend on where the camera ended up.** Safe ruling needs the on-screen size;
Solve stereo needs the distance to the glass and the case back. Reframe and they're
both stale. Any time you touch the lens, the camera distance, the case size, or the
render resolution — run both again. It's two clicks.

Terminal should show something like:

```
jg4d vitrine stereo: interaxial 8.8 mm (parallax-limited 8.8, 1/30 rule 18.3) |
converge 550 mm on the glass | back of case +1.00%, item +0.74% of width
  hfov 28.8 deg, lens 70 mm, res 4096x2160
  Nothing sits in front of the glass, so parallax is positive everywhere:
  no window violation is possible by construction.
```

**Sanity-check that:** interaxial should land ~8–20 mm. If it says 40 mm+ you've
got a doll's-house setup — move the camera back or shorten the case, then re-solve.
(The LDI3 player's 63 mm would give **+7.15%** here, about 7× your budget. That
default is right for a captured room and wrong for a vitrine.)

`Max parallax %` is your budget: **1.0 for anaglyph**. Polarised theatrical
tolerates 2%+, but red-cyan doesn't — cross-talk turns big parallax into ghosting.

---

## 7. Render settings

**Render Properties:**

- Engine: **Cycles** (Build sets this)
- Device: **GPU Compute** if you have it
- **Sampling > Render > Max Samples: 512–1024.** Volumetrics are noisy.
- **Light Paths > Volume: ≥2** (Build sets this)

**On denoising — this is the stereo-specific trap.** Left and right denoise
*independently*, so the denoiser invents slightly different detail in each eye.
Your visual system reads that mismatch as depth noise, and red-cyan is the least
forgiving format for it. Give it enough samples that the denoiser is only
polishing. If you see the volume sparkle or crawl in the anaglyph, that's this —
raise samples, don't raise denoising.

Then:

- **Output Properties > Stereoscopy** is already on
- **Format: Individual** ← for delivery
- Output: **PNG, RGB, 16-bit** (or OpenEXR)

`Solve stereo` leaves it on **Stereo 3D / anaglyph** so you get a red-cyan preview
in the viewport and the render window. **That preview is not Dubois** — it's
Blender's plain matrix. Switch to **Individual** before you render anything you
intend to keep.

Test render: **F12**. Look for the light cone in the haze, motes at their own
depths, and the sheen sweeping the scuffs.

---

## 8. Delivery

`Individual` writes `..._L.png` / `..._R.png`. Two options:

**Your existing chain:** grade L/R in Resolve, Dubois at the end — same as the
R5C/VVE material. This is the one to use; it keeps the vitrine inserts and the
captured footage in one grade.

**Or the repo's own operator** for quick checks: `N > jg4d > jg4d LDI3 player >
Dubois anaglyph from L/R pair`, pick the `_L` file. Same matrices as your web
player, so previews match.

Either way: **grade first, Dubois last.** Dubois is a lossy projection into
red-cyan; grading after it fights an already-collapsed image.

---

## 9. Randomize

Each button is `seed + 1` and rebuild that part — reproducible, and independent of
the others.

| button | changes |
|---|---|
| **Curl** | bend axis, bow, corner lift, twist, ripple |
| **Glass** | wipe arcs, scratches, smudges (procedural, instant) |
| **Dust** | mote distribution |
| **Lighting** | position, energy, Kelvin — bounded by **Light variation** |

**Light variation** defaults to **0.25** because you asked for *slightly*. At 0.25
the key moves a few cm and shifts ~200K. Push to 0.6+ only if you want shot-to-shot
variety that reads as different cases.

Curl and Glass are the two that change the shot most. Hit Curl a few times with
the viewport in Rendered and watch the sheen move across the sheet — that's the
sheet becoming an artifact instead of a billboard.

---

## 10. Then batch

Once one page looks right:

```bash
python3 tools/vitrine_prep.py ~/path/to/The_Farm_Book.pdf \
    --pages 12,20,44-52 -o ~/vitrine_items --dpi 300 --target-height 2160

# web-sourced items, each at its real size
python3 tools/vitrine_prep.py ~/Downloads/snapshot.jpg -o ~/vitrine_items --width-mm 89
python3 tools/vitrine_prep.py ~/Downloads/postcard.png -o ~/vitrine_items --width-mm 140
```

**`--width-mm` matters for anything not letter-size.** It sets the real-world size
in Blender, which drives dust scale, depth of field and the stereo solve. A
89 mm snapshot told it's 216 mm wide converges wrong and the dust looks like snow.

In Blender: point the file field at a different `_albedo.png` → **Build vitrine** →
**Safe ruling** → **Solve stereo**. Your framing survives a rebuild (`Reset camera
on build` is off by default).

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| Build errors, socket name in the message | Principled changed sockets in 4.0 | tell me the name; `set_input()` takes a fallback list |
| Everything black in Rendered view | working as designed | look through the camera (Numpad 0); check key energy |
| Can't see any operator output | macOS has no system console | launch Blender from Terminal (step 3) |
| Item looks like a flat sticker | curl at 0, or emission somewhere | raise **Curl**; the paper must be lit, never emissive |
| Dots invisible | ruling too fine, or relief too low | **Safe ruling**, then raise **Dot relief** (not Dot in colour) |
| Dots shimmer on a move | ruling still too fine | lower ruling below the safe value |
| Glass invisible | nothing for it to reflect | raise **Sheen W**, or set a **Room HDRI** |
| Volume sparkles / crawls in 3D | per-eye denoise mismatch | raise samples |
| Ghosting / eyes fight | parallax over budget | **Max parallax % → 0.8**, re-**Solve stereo** |
| Reads as a doll's house | interaxial too big for the subject | camera back or case shorter, re-solve |
| Screen survives the descreen | curled page, drifting frequency | `--wedge-deg 45 --drift 0.08` |
| Item flattened *into* the page | deep vignette, ambiguous | `--flatten 0.5` |
| Scan reports 8-bit but you scanned 16 | no opencv | `pip install opencv-python-headless` |

---

## The five things that actually bite

1. **Set render resolution before Build.** Safe ruling reads it.
2. **Re-run Safe ruling + Solve stereo after any reframe.** Both are geometry-dependent.
3. **Launch Blender from Terminal** or you never see the diagnostics.
4. **Switch views_format to Individual before rendering for real.** The anaglyph
   preview is not Dubois.
5. **`--width-mm` on anything that isn't a letter-size page.**
