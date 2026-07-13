# jg4d LDI3 player for Blender

Imports Lifecast LDI3 volumetric photos/video as **three real displaced layer
meshes**. Cycles/Eevee then handle occlusion, stereo, and lighting natively —
CG objects placed in the scene are correctly occluded by LDI foreground and
correctly occlude LDI background, per-pixel, both engines.

## Install

Edit > Preferences > Add-ons > Install... > `jg4d_ldi3_player.py`, enable
"jg4d LDI3 player". Blender 3.0+ (tested 3.0; written for 4.x too).

## Use

1. Convert your LDI3 video to frames (frame-perfect by construction):
   `ffmpeg -i uno.mp4 uno_ldi3_%06d.png`
   (or keep VVE's pre-encode `ldi3_%06d.png` intermediates — same thing,
   16-bit clean, no codec round-trip.)
2. 3D View > Sidebar (N) > jg4d > **Import LDI3**. Pick any frame of the
   sequence (a lone `.png`/`.jpg` imports as a still).
3. Scene frame N shows sequence file N (cycling past the end). Scrub away.
4. **Setup stereo camera** enables Multi-View at the capture point with
   red-cyan anaglyph output (63 mm interaxial, off-axis convergence).

## Parameters / futureproofing

Decode constants live in `DEFAULTS` at the top of the file and are overridden
per-import by a JSON sidecar (`<file>.json` or `jg4d_sidecar.json` next to the
frames):

```json
{ "inv_depth_coef": 0.3, "ftheta_scale": 1.15, "ftheta_inflation": 3.0,
  "max_depth": 50.0, "decode_12bit": true, "grid_n": 256 }
```

New depth backends in VVE (RAFT / DA3 / FoundationStereo) don't change the
LDI3 container — different constants or future layout tweaks go in the
sidecar, not the code. After editing parameters use **Refresh depth**.

## How it works (~450 lines, on purpose)

* Mesh: equiangular dome grid per layer (port of `web/lifecast_res/Ldi3Mesh.js`).
* Depth: numpy decode of the depth cells (12-bit ECC identical to
  `LifecastVideoPlayerShaders11.js`), vertices displaced radially by
  `inv_depth_coef / inverse_depth`; a `frame_change_post` handler re-displaces
  on every frame change (cached; ~0.3 s per new 5760² frame, instant after).
* Shading: 8 shader nodes per layer — UV remap into the color cell → emission;
  alpha cell → transparency (Hashed blend in Eevee, Transparent BSDF in Cycles).
* No Geometry Nodes, no drivers, no dependencies beyond bundled numpy.

Capture-forward is +Y, up is +Z, meters at metric scale. The importer creates
an empty `jg4d_ldi3` parent — move/rotate that to place the whole capture.
