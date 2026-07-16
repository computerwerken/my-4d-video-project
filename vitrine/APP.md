# jg4d vitrine — the app

Drop in scans and PDFs, read the QC, pick a look from a contact sheet, batch out
L/R masters. Local web app, stdlib only, no build step, nothing to code-sign.

```
jg4d_vitrine.command      double-click this
jg4d_app.py               server + API
jg4d_ui.html              the page
jg4d_blender_job.py       runs inside blender --background
vitrine_prep.py           the prep engine (also still a CLI)
vitrine_kit.py            the Blender addon (the app registers it itself)
```

Put all six in one folder. Double-click `jg4d_vitrine.command`. A Terminal window
opens (that's the log — closing it quits the app) and your browser opens on
`http://127.0.0.1:8756/`.

First run, `.command` may need: right-click → **Open** → **Open** to get past
Gatekeeper. Or `chmod +x jg4d_vitrine.command` if double-click does nothing.

---

## What it does

**Setup panel** — checks Blender, the two scripts, and every Python dep on every
poll, and prints the exact fix command for whatever's missing. Start here; if
Blender is red, rendering won't work no matter what else you do.

**1 · Prep** — point Source at a PDF (`Pages: 20` or `3,7,20-24`), an image, a
https:// URL (posters and other online documents are fetched for you), or a
folder. Set **Real width mm** for anything that isn't a letter-size page. Prep.

Each item lands in the **Items** table with QC pills:

| pill | meaning |
|---|---|
| `downscale 1.53x` | good — you have pixels to spare |
| `upscaling 0.24x` | **red** — soft. UpscaleVideo.ai from your repo first |
| `screen 133 lpi @ 88% Nyquist` | the scan's own aliasing; >70% is why descreen matters |
| `descreened (wedge+lp)` | which strategy it picked, automatically |
| `8-bit source` | fine for web-sourced, thin if you scanned it yourself |
| `falloff 23%` | lighting gradient removed; >25% goes amber and warns |

**View** opens a **100% crop**. Use it. A downscaled preview hides exactly the
aliasing you're checking for, so it's the only honest look. Lattice gone? Text
still crisp?

**2 · Look** — case, lens, parallax, curl, lighting. Set **Res X/Y first**: the
halftone ruling is computed from the frame height, so changing resolution later
invalidates it.

**Contact sheet** — renders N seed variants per item at low samples, quarter size,
**as anaglyph** so you judge the actual 3D rather than a flat left eye. Put the
red-cyan glasses on for this bit. Click one to choose it. Select an item on the
Items tab first, or leave none selected to do all of them.

**Render** — full-quality **16-bit Individual L/R** into `<project>/renders`.

Every job runs in the background; the log streams to the panel and the Terminal.
**Cancel** kills the Blender child process.

---

## What it deliberately doesn't do

**It stops at L/R masters.** No Dubois, no .mov. Dubois is a lossy projection into
red-cyan; grading after it means fighting an already-collapsed image. Import the
pairs into Resolve, grade them next to your R5C/VVE footage, Dubois at the very
end — same chain as the rest of the film.

**It doesn't touch your Blender config.** Renders run
`--background --factory-startup` and register `vitrine_kit` directly rather than
installing an addon. Whatever's in your startup file — an addon mid-upgrade, a
stray unit setting — has no say in what a batch render looks like. Every job gets
the same empty room.

**It doesn't frame for you.** The camera is driven by Lens / Cam dist. For a shot
that needs real framing, use the addon in Blender proper (see SETUP.md) and let
the app handle the volume work.

---

## Per item, per seed, the app always does this

```
resolution  ->  Build  ->  Safe ruling  ->  Solve stereo  ->  render
```

Never skipped, never reordered, recomputed for **every seed**, because a rebuild
can move things. That ordering is the whole reason to have an app: it's the part
that's easy to get wrong by hand, and getting it wrong costs you aliased dots and
a 7× parallax budget. The log shows the numbers for each:

```
blender: seed 3: ruling 65.5 lpi, interaxial 8.82 mm, converge 550.0 mm
```

Sanity-check interaxial: **8–20 mm**. 40 mm+ means the case reads as a doll's
house — move the camera back or shorten the case.

---

## Settings that matter

| | |
|---|---|
| **Res X/Y** | set before rendering. Drives the halftone ruling. 4096×2160 DCI 4K |
| **Max parallax %** | **1.0 for anaglyph**. Polarised tolerates 2%+; red-cyan doesn't |
| **Real width mm** | anything not letter-size. Drives dust scale, DOF, stereo |
| **Light variation** | 0.25 default — deliberately small |
| **Samples** | contact 48 is plenty. Final 512+; volumetrics are noisy |

**On samples and stereo:** left and right denoise *independently*, so the denoiser
invents slightly different detail in each eye, and your visual system reads that
mismatch as depth noise. Red-cyan is the least forgiving format for it. If the
volume sparkles in the anaglyph, raise samples — don't raise denoising.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `.command` won't open | right-click → Open → Open. Or `chmod +x` it |
| blender pill red | install Blender, or `JG4D_BLENDER=/path/to/Blender python3 jg4d_app.py` |
| port in use | `JG4D_PORT=8760 python3 jg4d_app.py` |
| opencv pill amber | `pip3 install opencv-python-headless` — without it 16-bit RGB TIFFs read as 8-bit |
| "a job is already running" | one at a time by design. Cancel first |
| render fails instantly | read the log — Blender's own error is passed through verbatim |
| contact sheet very slow | drop Size % to 15, Samples to 32 |
| items vanished | they follow the project folder. Re-set it; state reloads from `jg4d_project.json` |

The project folder holds everything — maps, previews, contact sheets, renders, and
`jg4d_project.json` with your look and picks. Re-open the app and it reloads. Move
the folder and nothing breaks.

---

## Honestly: what's tested and what isn't

I could run the Python here but not Blender and not a browser, so:

**Tested, by running it:** the server and every endpoint; the prep pipeline
end-to-end through the API on the real Farm Book PDF; job state and progress;
project save/reload; the QC pill logic; the setup checker (it correctly reports a
missing Blender); the `/file` path jail (403 outside the project); clean failure
with a readable message when Blender is absent; the job script's spec parsing and
error paths; the launcher script.

**Not tested — first run is the real test:** the browser UI itself (layout, the
JS), and everything downstream of actually launching Blender. The Blender-side
logic is the same code path SETUP.md walks through by hand, so if that works, this
should. If it doesn't, the Terminal window has the verbatim error.
