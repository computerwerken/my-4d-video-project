# jg 4d player

Fullscreen LDI3 volumetric video/photo player with **Dubois red-cyan anaglyph** output and mouse navigation. Single HTML file, no build step.

## Run

Serve this folder over HTTP (required for video textures), e.g. from the repo root:

    python3 web/local_server.py     # then open http://localhost:8000/jg4dplayer/

Or any static server (`npx serve`, GitHub Pages, ...). Then drag & drop an `*_ldi3.mp4` or `*_ldi3.jpg` produced by the Volumetric Video Editor, or pass a URL: `index.html?src=https://.../video_ldi3.mp4`.

## Controls

drag = look · WASD + QE / scroll = move · Space = play/pause · F = fullscreen · R = reset · M = mute · 1/2/3 = toggle LDI layers. HUD: IPD slider, 12-bit depth toggle, half-res toggle, eye swap.

Wear red-cyan glasses, red over the **left** eye.

## Credits

LDI3 mesh + decode shaders adapted from Lifecast's MIT-licensed player (`/web/lifecast_res`). Anaglyph mixing per Eric Dubois' least-squares matrices. Icon: pixel cat.
