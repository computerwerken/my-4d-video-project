# jg4d vitrine kit — review handoff

Everything a reviewer needs: what this is, every decision and why, every bug found
and the measurement that found it, what's verified and what isn't, and where I
think the remaining risk is. Written to be adversarially reviewed — the "Where to
attack this" section at the end is the point.

---

## 1. Context

**Who / what.** jg (github: `computerwerken`) is making an archival documentary.
Footage is Canon R5C VR180 run through Lifecast's Volumetric Video Editor (VVE).
Archival stills are scanned/downloaded. Delivery is a **Dubois red-cyan anaglyph
.mov projected in a theatre**, graded in DaVinci Resolve.

**The repo.** `github.com/computerwerken/my-4d-video-project` — a fork of
`fbriggs/lifecast_public` (Lifecast shut down and open-sourced everything), 11
commits ahead. Dirs: `UnrealEngine5/ blender/ deprecated/ jg4dplayer/
lenovo_mirage/ lifecast_apps/ nerf/ web/`.

**The existing addon** — `blender/jg4d_ldi3_player.py`, authored "jg + claude",
~450 lines "on purpose". Imports Lifecast LDI3 volumetric video as three displaced
layer meshes so Cycles handles occlusion/stereo/lighting natively. Its house style
is load-bearing and the new code follows it:

- numpy only. **No Geometry Nodes, no drivers**, no deps beyond bundled numpy.
- geometry re-displaced by a `frame_change_post` handler
- params in a **JSON sidecar** next to the media, overriding a `DEFAULTS` dict
- N-panel tab `jg4d`; operators `jg4d.import_ldi3`, `jg4d.refresh`,
  `jg4d.setup_stereo`, `jg4d.dubois`
- `jg4d.setup_stereo` hardcodes **63 mm interocular**, OFFAXIS, convergence 2.0 m
- `jg4d.dubois` holds the DUBOIS_L/DUBOIS_R matrices and gamma-2.2 handling,
  matching the jg4dplayer web app

**The ask.** Make scanned archival items appear to float in a museum vitrine
behind scuffed-but-clean glass — black background, unique lighting, volumetric
presence (dust, glass). Reference: **Prototype (Blake Williams)**, which floats
Galveston stereographs in black space. Then: check a sample scan, build an
automated version of a previously-proposed 4-export Photoshop stage, build
`vitrine_kit.py` with randomize buttons (glass / paper curl / scuffs / dust /
lighting — "only slightly").

**Prior-thread advice being implemented** (jg pasted it in). Broadly good, but it
contained one instruction this work contradicts — see §4.1.

---

## 2. Decisions jg made (asked explicitly, not assumed)

| Question | Answer |
|---|---|
| Halftone handling | **Descreen + resynthesise procedurally in Blender** |
| What is an "item" | *"not just pages from this book, but some snippets from the pages of this book and other images from around the web etc"* — so inputs are heterogeneous and often 8-bit web JPEGs |
| Photoshop stage | **Headless Python + .jsx handoff** |
| App scope | **Prep + batch render** (drive Blender headless; stop at L/R) |
| App form | **Local web UI** |
| App per-item variation | **Contact sheet → pick a seed → full render** |

---

## 3. The sample scan — measured, not assumed

`The_Farm_Book.pdf` — 190,453,271 bytes, 108 pages, PDF 1.6, Creator "Canon",
CreationDate 2010-08-03. Embedded page rasters, not vector.

| property | measured | method |
|---|---|---|
| pixels | 2551×3300, 300 ppi, RGB, 8-bit, JPEG | `pdfimages -list` |
| physical | 215.98 × 279.4 mm | derived |
| **halftone** | **133 lpi, 2.26 px/cell, 88.4% of Nyquist** | tile-averaged FFT, radial-background spike detection |
| **lattice axes** | **46.4°/136.8° AND 15.7°/106.0° — two orthogonal pairs** | discovered late; see §5.9 |
| coherence | 0.87 (a real lattice) | fraction of strong-peak energy at r0 |
| clipping | 0.00% white, 0.03% black | histogram |
| illumination falloff | ~16–23% | robust quadratic paper fit |
| paper colour | RGB 0.798/0.764/0.671 (warm sepia) | 90th-pct masked mean |
| JPEG blockiness | 1.058 (edge-of-block/interior gradient ratio) | mild |
| effective scale @2160px | **1.53× downscale** | |

**Verdict: usable.** Resolution is genuinely fine. The halftone is the whole
problem and dictates the pipeline's shape.

---

## 4. The three technical arguments (the substance of the review)

### 4.1 The screen is already aliased — so don't rebuild it faithfully

133 lpi sampled at 300 ppi = 2.26 px/cell = **88.4% of Nyquist**. The visible beat
in the scan *is* aliasing, baked in at capture. Therefore:

**The prior thread's advice to "high-pass the scan's luminance for a paper-grain
bump map — this is the secret ingredient" is wrong for this material.** Paper tooth
is 20–50 µm; resolving it needs 1200+ ppi. At 300 ppi a luminance high-pass does
not contain paper fibre — it contains **the aliased printing screen**. Feeding that
to a bump map yields a vibrating lattice, which in red-cyan is ghosting bait and
per-eye noise.

So: notch the lattice out here, export clean ink **density**, and rebuild a
controllable halftone procedurally in the shader where Cycles' pixel filter
anti-aliases it properly. Paper tooth is synthesised, not sampled.

### 4.2 Safe ruling — rebuilding the true ruling recreates the aliasing

A 279 mm sheet filling a 2160 px frame gives 196 ppi of screen resolution.
Rebuilding the real 133 lpi = **1.48 px/cell** — worse than the scan. Below ~3
px/cell a screen stops being dots and becomes shimmer, and shimmer is per-eye
noise. `safe_ruling(item_mm, on_screen_px, px_per_cell=3.0)` returns **~65 lpi**
for this item at 4K. Verified to invert to exactly 3.0 px/cell.

Artistically this is also the better call: at ~65 lpi the dots read as *ink*
catching raking light rather than as mush.

### 4.3 Interaxial — 63 mm is wrong here by ~7×

The LDI3 player's 63 mm is human eye spacing: correct for a captured room, wrong
for a 260 mm case at 420 mm on a 70 mm lens. Parallax as a fraction of frame width:

```
p = I · (1/Zc − 1/Z) / (2·tan(hfov/2))
```

At that geometry, 4096×2160:

| interaxial | parallax at case back |
|---|---|
| 63 mm (LDI3 default) | **+7.15%** — ~7× a 1% anaglyph budget |
| **8.8 mm (solved)** | +1.00% |
| 18.3 mm (1/30 rule) | — |

`solve_interaxial` inverts the parallax equation for a target and takes
`min(parallax-limited, Zg/30)`. Convergence goes **on the glass**: the theatre
screen becomes the vitrine pane, the item sits behind it in positive parallax, the
scuffs land at zero parallax where they're sharpest, and no window violation is
possible **by construction** — nothing is in front of the glass, and the dust
drift amplitude is a hard bound (verified: 22.2 mm clearance across 2400 frames).

---

## 5. Bugs found, with the measurement that found each

All in `vitrine_prep.py` unless noted. Every one was found by running the code, not
by reading it.

**5.1 Overlap-add missing its synthesis window.** `acc += filter(block·win)` but
`wsum += win²`. Global gain error — output **~1.8× too bright**. Fixed: `acc +=
filt·win` (proper WOLA). *Detected: descreened output visibly brighter; mean gain
now 0.983.*

**5.2 Descreen padded only right/bottom.** A Hann window is zero at its own edges,
so the first/last half-tile had `wsum → 0` and normalisation divided ~0 by ~0.
Border ring mean **0.978 → 0.379**. Fixed: pad all four sides by `hop`, crop.
*Now: all four borders match the original within 0.0000.*

**5.3 Wedge band reached DC.** A fixed 0.05 cyc/px radial band is harmless around a
near-Nyquist screen at 0.44 but swallows the origin around a screen at 0.025 —
notching DC removes the image's mean. **Whole image darkened 0.61×.** Fixed:
`sig_r = min(drift, r0·0.30)`, hard floor `band *= (fr > r0·0.5)`, plus `m[c,c]=1`.
*Now: DC preserved to 1.0000× on both test images.*

**5.4 Illumination flattener ate the picture.** The original local-max field
followed whatever was under it, so on a page with a large dark photograph it dipped
inside the photo and "flattening" **lifted the photograph 1.31×**. It reported
**61–72% falloff on a synthetic perfectly flat image**. Four approaches were tried
and measured:

| approach | why it failed |
|---|---|
| local-max field | chases the subject (1.31× lift) |
| robust quadratic LSQ | a big centred dark rectangle **is** shaped like a vignette |
| asymmetric least squares (p=0.98) | overshoots — surface bows to 1.5 above the paper |
| global Otsu preselect | inverts past ~50% vignette: shaded corner paper is darker than lit centre ink |
| **local-max-in-neighbourhood + texture test + robust quadratic** | shipped |

The final insight: illumination and ink differ in **scale**, not brightness. Ask
each cell whether it's bright relative to a third of a frame around it *and* smooth
(paper is smooth; anything printed isn't). Results:

| input | before | after |
|---|---|---|
| flat image, 54% dark subject | 61–72% "falloff", 1.31× lift | **0.0% falloff, 1.0000× lift** |
| 25% vignette | 12.3% paper spread | **0.1%** |
| 45% vignette | 24.5% | **2.6%** |
| 60% vignette | 35.4% | 18.4% (partial, **warns**) |
| 75% vignette | 48.3% | 37.2% (partial, **warns**) |
| Farm Book photo region | 1.31× lift | **~1.00×** |

**5.5 `maxfilt` fallback was a sparse comb.** Strided offsets + `np.roll` produced
shifted copies at rows {16,20,24,28,32} instead of a contiguous 17-row window, and
wrapped around the array. Fixed: exact separable max + cv2/scipy paths.

**5.6 `maxfilt` downcast float64→float32 — this one is the nastiest.** Callers do
`score == maxfilt(score)` to find local maxima. A float64→float32→float64 round
trip perturbs the low bits, so that equality **silently never held** and **screen
detection returned nothing at all on any machine without scipy** (i.e. most
machines). It was *exposed* by fixing 5.5 — the broken comb had been accidentally
letting the equality pass. Fixed: don't downcast (cv2 handles CV_64F), and compare
with tolerance. *Detected: a PDF run printed "screen: none detected" on a page
known to have a 133 lpi screen.*

**5.7 16-bit input silently destroyed.** `Image.convert("RGB")`:

- 16-bit **gray** TIFF (`I;16`) → **3 unique levels out of 512** (Pillow assumes a
  0–255 range and clips)
- 16-bit **RGB** TIFF → opens with mode already `'RGB'`, i.e. **halved to 8-bit**

Neither raises. This directly contradicted the pipeline's own 16-bit premise and
the prior thread's "scan flat at 16-bit". Fixed: `load_rgb()` uses cv2
`IMREAD_UNCHANGED` first, numpy-direct for Pillow's integer modes. Plus
`_declared_depth()` reads the container's own header (PNG IHDR byte 24, TIFF tag
258) and **warns when the file claims >8 but Pillow could only give 8**, so the loss
is attributed to the missing dependency rather than to the scanner. *Verified:
512/512 levels through both 16-bit paths.*

**5.8 `_screen_axes` used the median radius, and the report disagreed with the
decision.** The QC line said "12.1% of Nyquist" while the descreen mode was chosen
on 73.3% — two different peaks. Worse, the median of scattered peaks lands at a
frequency where no actual peak lives, so the wedge notched empty spectrum. Fixed:
**r0 = the strongest peak's radius** (that's what a fundamental *is*); angles =
strong peaks within ±15% of r0; added a **coherence** metric (fraction of strong
energy agreeing on r0) so scattered periodic hash is reported as such rather than
pretended to be a screen. *Farm Book coherence 0.87; a degenerate downscaled image
0.19 and flagged.*

**5.9 Per-tile refinement re-derived the axis set — and revealed a second
lattice.** Fixing 5.8 exposed that Farm Book page 20 has **two orthogonal halftone
lattices at the same frequency**: 46.4°/136.8° and 15.7°/106.0° — the photograph
and the tinted text block were screened separately. Per-tile refinement was
re-deriving the axis *set* locally and dropping one. The set is a global property
of the page; only the frequency drifts (curl/perspective). Fixed: pass global axes
into `build_mask`; local refine tracks frequency only. **Suppression 79% → 90%.**

**5.10 16-bit RGB PNG writing.** Pillow has no 16-bit RGB mode and
`frombytes('RGB;16B')` silently truncates. Hand-rolled a ~15-line zlib PNG encoder.
*Verified: ImageMagick reports "Depth: 16-bit"; an independent decode finds 61,193
unique R values (8-bit caps at 256).* Note Pillow **reads** them back as 8-bit —
a Pillow limitation, not the file; Blender and Photoshop are fine.

### `vitrine_kit.py` — found by review, not runtime (no Blender available)

- `hfov_of` had `sensor_fit AUTO` wrong: AUTO maps `sensor_width` to the **larger**
  pixel dimension; the code used `sensor_height`. This scales every parallax number.
  Fixed via `sensor_extents()`.
- Materials accumulated `.001/.002` on rebuild, and a lookup by name then fetched
  the stale one → `fresh_mat()`.
- `"Transmission Color"` doesn't exist on Principled in 3.x **or** 4.x — with
  Transmission=1, Base Color *is* the tint. Was failing silently.
- Halftone dot Fac double-multiplied `halftone_amount`.
- Glass box had no real UVs (painted scuff override would land on garbage) →
  `box_uvs()`.
- Glass procedural coords weren't normalised to pane size → wear changed scale when
  the case was resized.
- `clear_rig` destroyed the camera on rebuild (losing framing) and contained dead
  code → `keep=("camera",)` + a `reset_camera` prop.
- **Dust drift amplitude wasn't a hard bound**: 4 mm requested → 5.36 mm actual.
  Since "dust stays behind the glass" is the no-window-violation guarantee, this
  mattered. Normalised → exactly 4.000 mm.
- **Dust moved as a rigid lump**: spatial frequency ~9 rad/m meant sub-radian
  variation across the case. Raised to 110/85 rad/m. Correlation now 0.374 at
  <30 mm apart, decaying to 0.005 at >150 mm — a cloud, not a lump.
- Camera must exist *before* the paper material (safe ruling feeds the shader), and
  resolution before build.

---

## 6. Files

| file | lines | md5 | what |
|---|---|---|---|
| `vitrine_prep.py` | 1247 | `b271fde1972e75accd498844e97a1b9b` | prep engine + CLI |
| `vitrine_kit.py` | 1674 | `6b6e2ebadf8574c122c8fd53b09a01e5` | Blender addon |
| `jg4d_app.py` | 683 | `ba8cec64c4a883b6549c0dd9e1df8c30` | app server + API |
| `jg4d_ui.html` | 368 | `7c744e6accaf3d442beb16baf8d73a8a` | the page |
| `jg4d_blender_job.py` | 195 | `9ad24b16a70ea12a258a07fafc5698f5` | runs inside `blender --background` |
| `vitrine_prep_open.jsx` | 263 | | Photoshop round-trip |
| `jg4d_vitrine.command` | 30 | | launcher |
| `README_vitrine.md` / `SETUP.md` / `APP.md` | 195/318/153 | | docs |

**Data flow**

```
scan / PDF / web jpeg
  │  vitrine_prep.py  (numpy+Pillow; cv2/scipy/poppler optional)
  ▼
<stem>_albedo.png  16-bit sRGB   descreened, illumination-flattened
<stem>_alpha.png   16-bit gray   matte
<stem>_ink.png     16-bit gray   ink density -> dots + bump + gloss
<stem>_scuff.png   16-bit gray   glass wear (optional; addon is procedural)
<stem>.json                      analysis + real physical size
  │  vitrine_kit.py  (N-panel, or driven headless by the app)
  ▼
Cycles multiview Individual L/R 16-bit
  │  Resolve grade -> Dubois LAST
  ▼
anaglyph .mov
```

**Sidecar contract** — `vitrine_prep` writes, `vitrine_kit` reads: `physical_mm`,
`paper_rgb`, `resynth_halftone{ruling_lpi,angle_deg}`, `screen{...}`, `ppi`,
`pixels`, `matte_mode`, `source_bit_depth`, `descreened`, `illumination_falloff`,
`clipping_pct`, `effective_scale_at_target`. Given `<stem>_albedo.png` the addon
finds all siblings automatically.

**Key entry points**

- prep: `detect_screen` → `_screen_axes` → `build_mask` → `descreen` →
  `make_matte` → `flatten_illumination` → `make_ink_map` → `make_scuff_map`
- addon ops: `jg4d.vitrine_build`, `.vitrine_safe_ruling`, `.vitrine_solve_stereo`,
  `.vitrine_rand_{curl,scuff,dust,light,all}`
- addon: 8 operators, 48 props, 11 registered classes

**Invariant the app enforces per item per seed:**
`resolution → Build → Safe ruling → Solve stereo → render`. Both solves are
geometry-dependent and go stale on any reframe.

---

## 7. Verification status — read this before trusting anything

### Verified by execution

**Prep** (on the real `The_Farm_Book.pdf` and synthetics):
- illumination: invents nothing on a flat image (0.0%, 1.0000× lift); 25%→0.1%,
  45%→2.6%, 60%→18.4%, 75%→37.2% (last two warn)
- descreen: DC preserved 1.0000×, all four borders clean, 90% lattice suppression,
  report == decision, coherence 0.87 vs 0.19 on degenerate input
- matte: white ground knocked out; full page → no cut
- bit depth: 16-bit gray/RGB TIFF → 512/512 levels; 16-bit PNG output confirmed by
  ImageMagick + independent decode
- runs on bare numpy+Pillow (cv2/scipy/poppler blocked) — degrades, doesn't crash
- clean-room run from the 190 MB PDF, fresh dir

**Addon core, under a mock `bpy`** (numpy/math only):
- `solve_interaxial` inverts `parallax_frac` exactly at 0.5/1.0/2.0% targets
- convergence plane = exactly 0.000% parallax; in-front = negative (sign convention)
- `safe_ruling` inverts to exactly 3.0 px/cell
- `paper_curl` deterministic per seed; amplitude a hard bound; curl=0 exactly flat
- dust seed-deterministic; drift a hard bound (4.000 mm for amp 4.0); **22.2 mm
  clearance behind the glass across 2400 frames**
- `kelvin_to_rgb` monotone warm→cool, in gamut 1000–12000 K
- sidecar round-trip resolves all four maps + physical size
- self-consistency: 0 operators referenced-but-undefined, 0 unknown props in panel

**App:** server + every endpoint; full prep through the API on the real PDF; job
state/progress; project save/reload; QC pill logic; setup checker (correctly
reports missing Blender); `/file` path jail (403 on `/etc/passwd`); clean readable
failure when Blender is absent; job-script spec parsing (file + inline JSON) and
error paths; `bash -n` on the launcher.

### NOT verified — the honest list

- **Blender never ran.** `pip install bpy` has no distribution for this sandbox's
  platform/Python. **No node graph has ever been evaluated. Nothing has been
  rendered.** Every `vitrine_kit.py` fix above came from reading, not running.
- **The browser UI never rendered.** No display. Layout and JS are unexercised.
- **Everything downstream of launching Blender** in the app.
- The halftone resynthesis has never been *seen* — only argued for.
- No test on jg's other material (web-sourced snippets), only the Farm Book and
  synthetics.

---

## 8. Where to attack this

Ranked by where I think the risk actually is.

1. **Blender API compatibility across 3.0 / 4.0 / 4.2+.** The highest-risk surface,
   entirely unexercised. Principled sockets renamed in 4.0 (`Transmission` →
   `Transmission Weight`, etc.) — handled by `set_input()`'s fallback-name list, but
   that list is from memory, not from running 4.x. `blend_method`/`shadow_method`
   vanished in 4.2 (guarded by `hasattr`). `ShaderNodeMixRGB` is legacy-but-present
   in 4.x — is it? `Mesh.shade_smooth()` is 4.1+ (guarded). Check every
   `_n(nt, "ShaderNode...")` and every `set_input` name list.
2. **`bpy.ops` in `--background`.** All operators use `context.scene` and should be
   fine headless, but this is asserted, not observed. `jg4d_blender_job.py` also
   calls `vitrine_kit.register()` directly under `--factory-startup`.
3. **The halftone resynth's actual appearance.** The sine-product screen
   (`0.5+0.5·sin(2πnu)·sin(2πnv)` thresholded against ink density) is textbook AM
   halftone, but it has never been rendered. Does it read as ink or as a pattern?
   Does `Dot relief` at 0.55 do anything visible?
4. **1% parallax as the anaglyph budget.** This is my number, from general
   stereographic practice, not from testing on jg's screen. Cross-talk depends on
   the projector and the glasses. Worth a real-world check.
5. **The descreen on non-Farm-Book material.** Tuned and measured on one page plus
   synthetics. Genuinely periodic subjects (textiles, brick, guilloche) will look
   like screens to the detector — `--no-descreen` exists but detection of the
   pathology is only via the coherence number.
6. **The illumination fit on real deep-vignette scans.** >25% is honestly partial.
   The local-max-window size (`gy//3`) and the 0.85 threshold and the 45th-pct
   texture cut are all tuned on synthetics.
7. **`_declared_depth` / `load_rgb` on real scanner output.** Tested on TIFFs I
   generated with cv2, not on files from an actual Epson/flatbed.
8. **Concurrency in the app.** One job at a time by design (`JOB.start` guard), but
   `PROJ.items` is mutated from worker threads under a coarse lock; the state dict
   is also serialised on every item.
9. **The `.command` launcher on a clean macOS.** Gatekeeper, `pip install --user`
   into a system Python, Homebrew vs python.org Python.

### Design choices worth challenging (defensible, not obvious)

- **Stopping at L/R** rather than delivering a .mov. Rationale: Dubois is a lossy
  projection; grading after it fights a collapsed image.
- **`--factory-startup`** for batch renders: reproducibility over convenience.
- **No Geometry Nodes / no drivers** — inherited from the LDI3 player's stated
  philosophy, at some cost (dust is a hand-built mesh + a frame handler).
- **Descreen + resynth** rather than preserving the original dots. jg chose this,
  but the alternative (accept the aliasing, hold items static and large) is
  defensible for an archival film that wants the artifact's real texture.
- **`safe_ruling` at 3 px/cell.** Why 3? It's a judgement, not a theorem.
- **Reporting partial corrections + warning** rather than either refusing or
  silently doing damage.

### Known limits (documented, accepted)

- Deep vignettes >25% → partial correction, warned
- Genuinely periodic subjects → false-positive screens; use `--no-descreen`
- Pillow reads 16-bit PNGs back as 8-bit (Pillow's limit, not the file)
- Without opencv, 16-bit RGB TIFF halved (now warns loudly)
- Volumetrics: L/R denoise independently; mismatched noise is per-eye noise and
  red-cyan is the least forgiving format for it. Mitigation is samples, not
  denoising. Untested at scale.

---

## 9. One-line summary for the reviewer

A two-stage pipeline (headless Python prep → Blender addon → optional local web
app) that floats scanned archival material in a museum vitrine for red-cyan
anaglyph theatrical delivery; the prep engine is heavily tested and had 10 real
bugs found by measurement, while **the entire Blender half has never been
executed** and is the place to look hardest.


---

## 10. Review round 2 (fresh thread, executed this time)

**The §8 risk list was attacked in order. Headline: the Blender half now HAS
been executed** — Blender 3.0.1 headless (deb-extracted, arm64): register,
vitrine_build, safe_ruling, solve_stereo, and real Cycles renders in both
modes (contact anaglyph PNG; final Individual L/R). Render is visually sane
(poster legible, red/cyan separation present). Solve output at 512x288:
ruling 14.3 lpi, interaxial 5.65 mm, converge on glass at 420 mm — consistent
with §4's math scaled to the tiny test frame.

### Changed in this round

* **kit 1.1.0** — all 7 `ShaderNodeMixRGB` (legacy, §8.1's named 4.x risk)
  replaced with a `_mix_color()` helper: `ShaderNodeMix` (3.4+, color sockets
  BY INDEX 6/7/2 — the names collide) with MixRGB fallback. Regression: render
  statistics identical pre/post on 3.0 (mean 212.4/209.4/199.3).
* **prep: web-input round.** The user's ask — "not just PDFs but images and
  lots of other online documents (posters etc)":
  - `SUPPORTED_EXTS` single source of truth (prep, app worker, /api/browse);
    adds bmp/dib/gif/jp2/jfif + avif/heic via optional pillow-heif.
  - **https:// URLs** accepted by CLI, API and UI (`fetch_url` -> work dir,
    Content-Type fallback for extension). Verified against a loopback server.
  - **72/96 dpi metadata ignored** (software defaults, not measurements — a
    900 px download was becoming a 318 mm sheet). No trustworthy size ->
    assume 300 ppi, print the warning, set `size_assumed` in sidecar + QC pill.
  - **`--preset` physical sizes** (poster/onesheet/a2/a3/a4/letter/tabloid/
    postcard/photo/ticket/card) in CLI + UI dropdown.
  - **Source alpha becomes the matte** (`load_rgba`; matte_mode "source").
    Transparent-pixel RGB is rebuilt from opaque neighbours
    (`fill_under_alpha`) so analysis and texture taps never see the undefined
    black — this was measured live: a transparent clipout read as "51.96%%
    black clipping" and only matted correctly by luck.
  - **Coherence gate acts** instead of just warning: coherence < 0.5 in auto
    -> point-notches only, no wedge (measured on a synthetic text poster:
    wedge smeared content at 73%% "suppression"; notch-only 3%%).
  - **Designed-PDF fallback**: largest embedded image < 35%% of page area at
    150 dpi -> render the page (posters-as-PDF got a logo before).
  - **Stem dedup** (poster.bmp + poster.gif collided and overwrote).
* **app**: URL sources; preset passthrough; pillow-heif setup row; honest
  poppler message (PyMuPDF fallback exists); browse filter from SUPPORTED_EXTS.
  Smoke-tested end-to-end over HTTP: prep of a transparent PNG with
  preset=postcard -> matte "source", physical_mm [152,152], QC pills correct.

### Still not verified (inherited)

Blender 4.x specifically (3.0 executed; 4.x socket-name fallbacks remain
book-knowledge); the browser UI's rendering (JS unexercised beyond API
calls); halftone resynth appearance at real scale; the 1%% parallax budget on
the actual projector.
