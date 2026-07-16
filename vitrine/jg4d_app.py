#!/usr/bin/env python3
# jg4d vitrine - desktop app
# --------------------------
# Local web app: drop in scans/PDFs, prep them with a QC table you can actually
# read, render a contact sheet of seed variants per item, pick one, batch out the
# L/R masters for your Resolve -> Dubois chain.
#
#   python3 jg4d_app.py            (or double-click jg4d_vitrine.command)
#
# Stdlib only for the app itself -- http.server plus one HTML page. No Flask, no
# Electron, no build step, nothing to code-sign. The prep stage imports
# vitrine_prep (numpy + Pillow); the render stage shells out to Blender.
#
# Deliberately stops at L/R masters. Baking Dubois here would put a lossy
# projection in front of your grade, which is backwards.
#
# MIT License.

import base64
import glob
import html
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    sys.path.insert(0, HERE)
    from vitrine_prep import SUPPORTED_EXTS
except Exception:
    SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".tif", ".tiff", ".webp",
                      ".bmp", ".dib", ".gif", ".jp2", ".avif", ".heic", ".heif")
PORT = int(os.environ.get("JG4D_PORT", "8756"))
STATE_NAME = "jg4d_project.json"

sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------
# Setup discovery
# ----------------------------------------------------------------------------

def find_blender():
    env = os.environ.get("JG4D_BLENDER")
    if env and os.path.isfile(env):
        return env
    cands = []
    for pat in ("/Applications/Blender.app/Contents/MacOS/Blender",
                "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
                "/Applications/Blender*.app/Contents/MacOS/Blender",
                os.path.expanduser("~/Applications/Blender*.app/Contents/MacOS/Blender"),
                "/usr/share/blender/blender", "/opt/blender/blender"):
        cands.extend(sorted(glob.glob(pat), reverse=True))
    w = shutil.which("blender")
    if w:
        cands.append(w)
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def blender_version(path):
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
        m = re.search(r"Blender\s+([0-9]+\.[0-9]+(\.[0-9]+)?)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def check_setup():
    s = {}
    try:
        import numpy
        s["numpy"] = {"ok": True, "detail": numpy.__version__}
    except Exception as e:
        s["numpy"] = {"ok": False, "detail": str(e), "fix": "pip3 install numpy"}
    try:
        import PIL
        s["pillow"] = {"ok": True, "detail": PIL.__version__}
    except Exception as e:
        s["pillow"] = {"ok": False, "detail": str(e), "fix": "pip3 install pillow"}
    try:
        import cv2
        s["opencv"] = {"ok": True, "detail": cv2.__version__}
    except Exception:
        s["opencv"] = {"ok": False, "warn": True,
                       "detail": "missing - 16-bit RGB TIFFs will be read as 8-bit",
                       "fix": "pip3 install opencv-python-headless"}
    try:
        import scipy
        s["scipy"] = {"ok": True, "detail": scipy.__version__}
    except Exception:
        s["scipy"] = {"ok": False, "warn": True, "detail": "missing - prep ~2x slower",
                      "fix": "pip3 install scipy"}
    s["poppler"] = ({"ok": True, "detail": shutil.which("pdfimages")}
                    if shutil.which("pdfimages") else
                    {"ok": False, "warn": True,
                     "detail": "missing - PDFs fall back to PyMuPDF (pip3 install pymupdf)",
                     "fix": "brew install poppler"})
    try:
        import pillow_heif
        s["pillow_heif"] = {"ok": True, "detail": getattr(pillow_heif, "__version__", "?")}
    except Exception:
        s["pillow_heif"] = {"ok": False, "warn": True,
                            "detail": "missing - no HEIC/AVIF input (web posters)",
                            "fix": "pip3 install pillow-heif"}
    prep = os.path.join(HERE, "vitrine_prep.py")
    s["vitrine_prep"] = ({"ok": True, "detail": prep} if os.path.isfile(prep)
                         else {"ok": False, "detail": "not next to this app",
                               "fix": "put vitrine_prep.py in %s" % HERE})
    kit = find_kit()
    s["vitrine_kit"] = ({"ok": True, "detail": kit} if kit
                        else {"ok": False, "detail": "not found",
                              "fix": "put vitrine_kit.py next to this app, or in ../blender/"})
    b = find_blender()
    if b:
        v = blender_version(b)
        okv = True
        if v:
            try:
                okv = tuple(int(x) for x in v.split(".")[:2]) >= (3, 0)
            except Exception:
                okv = True
        s["blender"] = {"ok": okv, "detail": "%s  (%s)" % (b, v or "version unknown"),
                        "fix": None if okv else "vitrine_kit needs Blender 3.0+"}
    else:
        s["blender"] = {"ok": False, "detail": "not found",
                        "fix": "install Blender, or set JG4D_BLENDER=/path/to/Blender"}
    return s


def find_kit():
    for p in (os.path.join(HERE, "vitrine_kit.py"),
              os.path.join(HERE, "blender", "vitrine_kit.py"),
              os.path.join(os.path.dirname(HERE), "blender", "vitrine_kit.py")):
        if os.path.isfile(p):
            return p
    return None


# ----------------------------------------------------------------------------
# Project state
# ----------------------------------------------------------------------------

DEFAULT_LOOK = {
    "case_width_mm": 340.0, "case_height_mm": 420.0, "case_depth_mm": 260.0,
    "item_depth_mm": 40.0, "item_tilt": 8.0, "curl_mm": 3.0,
    "lens_mm": 70.0, "cam_distance_mm": 420.0, "target_parallax": 1.0,
    "volume_density": 0.012, "dust_count": 400,
    "key_energy": 12.0, "fill_energy": 1.2, "sheen_energy": 6.0,
    "var_light": 0.25, "scuff_amount": 0.7,
}


class Project:
    def __init__(self):
        self.out_dir = os.path.expanduser("~/vitrine_items")
        self.items = {}          # stem -> {...}
        self.look = dict(DEFAULT_LOOK)
        self.res_x, self.res_y = 4096, 2160
        self.lock = threading.Lock()

    def path(self):
        return os.path.join(self.out_dir, STATE_NAME)

    def save(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(self.path(), "w") as f:
                json.dump({"items": self.items, "look": self.look,
                           "res_x": self.res_x, "res_y": self.res_y}, f, indent=2)
        except Exception as e:
            LOG.put("could not save project: %s" % e)

    def load(self):
        p = self.path()
        if not os.path.isfile(p):
            return
        try:
            with open(p) as f:
                d = json.load(f)
            self.items = d.get("items", {})
            self.look.update(d.get("look", {}))
            self.res_x = d.get("res_x", self.res_x)
            self.res_y = d.get("res_y", self.res_y)
            LOG.put("loaded project: %d item(s)" % len(self.items))
        except Exception as e:
            LOG.put("could not load project: %s" % e)


PROJ = Project()


class Log:
    def __init__(self, n=4000):
        self.lines = []
        self.n = n
        self.lock = threading.Lock()

    def put(self, msg):
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > self.n:
                self.lines = self.lines[-self.n:]
        print(line, flush=True)

    def since(self, i):
        with self.lock:
            return self.lines[max(0, i):], len(self.lines)


LOG = Log()


class Job:
    def __init__(self):
        self.active = False
        self.kind = ""
        self.done = 0
        self.total = 0
        self.msg = ""
        self.cancel = False
        self.proc = None
        self.lock = threading.Lock()

    def start(self, kind, total):
        with self.lock:
            if self.active:
                return False
            self.active, self.kind, self.done, self.total = True, kind, 0, total
            self.msg, self.cancel, self.proc = "starting", False, None
            return True

    def stop(self):
        with self.lock:
            self.active = False
            self.proc = None

    def snap(self):
        with self.lock:
            return {"active": self.active, "kind": self.kind, "done": self.done,
                    "total": self.total, "msg": self.msg}


JOB = Job()


# ----------------------------------------------------------------------------
# Prep
# ----------------------------------------------------------------------------

def make_previews(stem, out_dir):
    """A fit-to-width view and a 100% centre crop.

    The crop is the point. Judging a descreen from a downscaled preview is
    meaningless -- the downscale hides exactly the aliasing you are checking for.
    """
    from PIL import Image
    import numpy as np
    sys.path.insert(0, HERE)
    from vitrine_prep import load_rgb

    pdir = os.path.join(out_dir, ".previews")
    os.makedirs(pdir, exist_ok=True)
    alb = os.path.join(out_dir, stem + "_albedo.png")
    if not os.path.isfile(alb):
        return None, None
    rgb, _ = load_rgb(alb)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8"))
    w, h = im.size

    full = os.path.join(pdir, stem + "_view.jpg")
    im.copy().resize((900, max(1, int(900 * h / w))), Image.LANCZOS).save(full, quality=88)

    c = 620
    x0, y0 = max(0, w // 2 - c // 2), max(0, int(h * 0.62) - c // 2)
    crop = os.path.join(pdir, stem + "_crop.jpg")
    im.crop((x0, y0, min(w, x0 + c), min(h, y0 + c))).save(crop, quality=94)
    return full, crop


def prep_worker(spec):
    try:
        import argparse
        from vitrine_prep import process, pdf_pages, parse_pages

        src = spec["source"]
        out = PROJ.out_dir
        os.makedirs(out, exist_ok=True)
        work = os.path.join(out, "_work")
        os.makedirs(work, exist_ok=True)

        args = argparse.Namespace(
            dpi=spec.get("dpi") or None,
            width_mm=spec.get("width_mm") or None,
            target_height=int(spec.get("target_height") or 2160) or None,
            matte=spec.get("matte", "auto"), feather=1.2,
            no_descreen=bool(spec.get("no_descreen")), descreen=spec.get("descreen", "auto"),
            no_refine=False, screen_snr=9.0, notch_sigma=3.0,
            wedge_deg=float(spec.get("wedge_deg", 30.0)),
            drift=float(spec.get("drift", 0.05)),
            flatten=float(spec.get("flatten", 1.0)), neutralize=False,
            preset=(spec.get("preset") or None),
            ink_highpass_mm=6.0, seed=int(spec.get("seed", 0)),
            scuff_size=2048, scuff_intensity=1.0)

        if re.match(r"^https?://", src, re.I):
            from vitrine_prep import fetch_url
            LOG.put("fetching %s ..." % src)
            try:
                src = fetch_url(src, work)
            except Exception as e:
                LOG.put("fetch failed: %s" % e)
                return
        jobs = []
        if os.path.isdir(src):
            for f in sorted(os.listdir(src)):
                if f.lower().endswith(SUPPORTED_EXTS):
                    jobs.append((os.path.join(src, f), None, None))
        elif src.lower().endswith(".pdf"):
            base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(src))[0])
            LOG.put("extracting pages from %s ..." % os.path.basename(src))
            for p, path in pdf_pages(src, parse_pages(spec.get("pages") or None), work,
                                     args.dpi):
                jobs.append((path, "%s_p%03d" % (base, p), p))
        else:
            jobs.append((src, None, None))

        if not jobs:
            LOG.put("nothing to prep in %s" % src)
            return

        JOB.total = len(jobs)
        LOG.put("prepping %d item(s) -> %s" % (len(jobs), out))

        taken = set()
        for path, stem, page in jobs:
            if JOB.cancel:
                LOG.put("cancelled")
                break
            from vitrine_prep import stem_of
            stem = stem or stem_of(path, out, taken)
            name = stem
            JOB.msg = name
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                process(path, out, args, stem=stem, page=page)
            except Exception as e:
                sys.stdout = old
                LOG.put("FAILED %s: %s" % (name, e))
                LOG.put(traceback.format_exc().strip().splitlines()[-1])
                JOB.done += 1
                continue
            finally:
                sys.stdout = old
            for line in buf.getvalue().splitlines():
                if line.strip():
                    LOG.put(line.rstrip())

            real = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                          stem or os.path.splitext(os.path.basename(path))[0])
            side = os.path.join(out, real + ".json")
            meta = {}
            if os.path.isfile(side):
                with open(side) as f:
                    meta = json.load(f)
            try:
                make_previews(real, out)
            except Exception as e:
                LOG.put("preview failed for %s: %s" % (real, e))
            with PROJ.lock:
                PROJ.items[real] = {
                    "stem": real, "albedo": os.path.join(out, real + "_albedo.png"),
                    "meta": meta, "seed": None, "contact": [], "final": [],
                    "qc": qc_of(meta),
                }
            JOB.done += 1
            PROJ.save()
        LOG.put("prep finished (%d/%d)" % (JOB.done, JOB.total))
    except Exception:
        LOG.put("prep crashed: %s" % traceback.format_exc().strip().splitlines()[-1])
        for l in traceback.format_exc().splitlines():
            LOG.put("  " + l)
    finally:
        JOB.stop()


def qc_of(meta):
    """Turn the sidecar into the handful of judgements worth surfacing."""
    q = []
    scr = meta.get("screen")
    eff = meta.get("effective_scale_at_target")
    if eff is not None:
        if eff < 0.95:
            q.append(["bad", "upscaling %.2fx - soft; UpscaleVideo.ai first" % eff])
        elif eff < 1.15:
            q.append(["warn", "near 1:1 (%.2fx)" % eff])
        else:
            q.append(["ok", "downscale %.2fx" % eff])
    if meta.get("size_assumed"):
        q.append(["warn", "size assumed (300 ppi) - set width or preset"])
    if meta.get("source_bit_depth") == 8:
        q.append(["warn", "8-bit source"])
    elif meta.get("source_bit_depth"):
        q.append(["ok", "%d-bit" % meta["source_bit_depth"]])
    if isinstance(scr, dict) and scr.get("pct_of_nyquist"):
        lvl = "warn" if scr["pct_of_nyquist"] > 70 else "ok"
        q.append([lvl, "screen %.0f lpi @ %.0f%% Nyquist" % (scr["ruling_lpi"],
                                                             scr["pct_of_nyquist"])])
    d = meta.get("descreened")
    if d and d != "none":
        q.append(["ok", "descreened (%s)" % d])
    elif d == "none":
        q.append(["ok", "no screen"])
    f = meta.get("illumination_falloff")
    if f is not None:
        q.append((["warn", "falloff %.0f%% - ambiguous" % (f * 100)] if f > 0.25
                  else ["ok", "falloff %.0f%%" % (f * 100)]))
    c = meta.get("clipping_pct") or {}
    if (c.get("white", 0) > 1) or (c.get("black", 0) > 1):
        q.append(["warn", "clipping %.1f%%W %.1f%%B" % (c.get("white", 0), c.get("black", 0))])
    return q


# ----------------------------------------------------------------------------
# Blender
# ----------------------------------------------------------------------------

def run_blender(job_spec, on_line=None):
    b = find_blender()
    if not b:
        raise RuntimeError("Blender not found. Set JG4D_BLENDER=/path/to/Blender")
    script = os.path.join(HERE, "jg4d_blender_job.py")
    if not os.path.isfile(script):
        raise RuntimeError("jg4d_blender_job.py missing from %s" % HERE)

    spec_file = os.path.join(PROJ.out_dir, "_work", "job.json")
    os.makedirs(os.path.dirname(spec_file), exist_ok=True)
    with open(spec_file, "w") as f:
        json.dump(job_spec, f)

    cmd = [b, "--background", "--factory-startup", "--python", script, "--", spec_file]
    LOG.put("$ %s --background ... -- job.json" % os.path.basename(b))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    with JOB.lock:
        JOB.proc = p
    result, err = None, None
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("JG4D-RESULT: "):
            try:
                result = json.loads(line[len("JG4D-RESULT: "):])
            except Exception:
                pass
        elif line.startswith("JG4D-ERROR: "):
            err = line[len("JG4D-ERROR: "):]
            LOG.put("blender: " + err)
        elif line.startswith("JG4D: "):
            LOG.put("blender: " + line[6:])
            if on_line:
                on_line(line[6:])
        elif "Error" in line or "Traceback" in line or "error:" in line.lower():
            LOG.put("blender: " + line)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(err or "Blender exited %d (see log)" % p.returncode)
    return result or []


def render_worker(kind, stems, opts):
    try:
        kit = find_kit()
        if not kit:
            LOG.put("vitrine_kit.py not found; cannot render")
            return
        JOB.total = len(stems)
        for stem in stems:
            if JOB.cancel:
                LOG.put("cancelled")
                break
            it = PROJ.items.get(stem)
            if not it:
                continue
            JOB.msg = stem
            if kind == "contact":
                n = int(opts.get("variants", 4))
                seeds = list(range(1, n + 1))
                out = os.path.join(PROJ.out_dir, ".contact", stem)
                shutil.rmtree(out, ignore_errors=True)
                spec = {"kit_dir": os.path.dirname(kit), "albedo": it["albedo"],
                        "out": out, "mode": "contact", "stem": stem, "seeds": seeds,
                        "look": PROJ.look, "samples": int(opts.get("samples", 48)),
                        "res_x": PROJ.res_x, "res_y": PROJ.res_y,
                        "res_pct": int(opts.get("res_pct", 25)),
                        "device": opts.get("device", "GPU")}
                res = run_blender(spec)
                with PROJ.lock:
                    it["contact"] = res
            else:
                seed = it.get("seed") or 1
                out = os.path.join(PROJ.out_dir, "renders")
                spec = {"kit_dir": os.path.dirname(kit), "albedo": it["albedo"],
                        "out": out, "mode": "final", "stem": stem, "seeds": [seed],
                        "look": PROJ.look, "samples": int(opts.get("samples", 512)),
                        "res_x": PROJ.res_x, "res_y": PROJ.res_y, "res_pct": 100,
                        "device": opts.get("device", "GPU")}
                res = run_blender(spec)
                with PROJ.lock:
                    it["final"] = res
            JOB.done += 1
            PROJ.save()
        LOG.put("%s finished (%d/%d)" % (kind, JOB.done, JOB.total))
    except Exception as e:
        LOG.put("%s failed: %s" % (kind, e))
    finally:
        JOB.stop()


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except Exception:
            return {}

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200, UI, "text/html; charset=utf-8")
        if u.path == "/api/state":
            with PROJ.lock:
                items = [dict(v, meta_small={
                    "physical_mm": v["meta"].get("physical_mm"),
                    "pixels": v["meta"].get("pixels"),
                }) for v in PROJ.items.values()]
            return self._send(200, {
                "setup": check_setup(), "out_dir": PROJ.out_dir, "look": PROJ.look,
                "res_x": PROJ.res_x, "res_y": PROJ.res_y,
                "items": items, "job": JOB.snap(),
            })
        if u.path == "/api/log":
            lines, n = LOG.since(int(q.get("since", ["0"])[0]))
            return self._send(200, {"lines": lines, "n": n})
        if u.path == "/file":
            p = q.get("p", [""])[0]
            if not p or not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
            # only ever serve from inside the project folder
            if not os.path.abspath(p).startswith(os.path.abspath(PROJ.out_dir)):
                return self._send(403, {"error": "outside project"})
            ctype = ("image/jpeg" if p.lower().endswith((".jpg", ".jpeg"))
                     else "image/png" if p.lower().endswith(".png") else "application/octet-stream")
            with open(p, "rb") as f:
                return self._send(200, f.read(), ctype)
        if u.path == "/api/browse":
            base = q.get("p", [os.path.expanduser("~")])[0]
            base = os.path.abspath(os.path.expanduser(base))
            if not os.path.isdir(base):
                base = os.path.dirname(base) or os.path.expanduser("~")
            try:
                names = sorted(os.listdir(base))
            except Exception as e:
                return self._send(200, {"path": base, "entries": [], "error": str(e)})
            ents = []
            for nm in names:
                if nm.startswith("."):
                    continue
                fp = os.path.join(base, nm)
                isd = os.path.isdir(fp)
                if isd or nm.lower().endswith(SUPPORTED_EXTS + (".pdf",)):
                    ents.append({"name": nm, "path": fp, "dir": isd})
            return self._send(200, {"path": base, "parent": os.path.dirname(base),
                                    "entries": ents[:600]})
        return self._send(404, {"error": "no route"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        b = self._body()
        if u.path == "/api/outdir":
            p = os.path.abspath(os.path.expanduser(b.get("path", "")))
            if not p:
                return self._send(400, {"error": "no path"})
            PROJ.out_dir = p
            PROJ.items = {}
            PROJ.load()
            LOG.put("project folder: %s" % p)
            return self._send(200, {"ok": True})
        if u.path == "/api/prep":
            src = os.path.expanduser(b.get("source", "").strip())
            if not re.match(r"^https?://", src, re.I) and not os.path.exists(src):
                return self._send(400, {"error": "no such path: %s" % src})
            b["source"] = src
            if not JOB.start("prep", 1):
                return self._send(409, {"error": "a job is already running"})
            threading.Thread(target=prep_worker, args=(b,), daemon=True).start()
            return self._send(200, {"ok": True})
        if u.path == "/api/look":
            PROJ.look.update(b.get("look", {}))
            PROJ.res_x = int(b.get("res_x", PROJ.res_x))
            PROJ.res_y = int(b.get("res_y", PROJ.res_y))
            PROJ.save()
            return self._send(200, {"ok": True})
        if u.path in ("/api/contact", "/api/render"):
            kind = "contact" if u.path.endswith("contact") else "final"
            stems = b.get("stems") or list(PROJ.items)
            if kind == "final":
                stems = [s for s in stems if PROJ.items.get(s, {}).get("seed")]
                if not stems:
                    return self._send(400, {"error": "pick a variant for at least one item first"})
            if not stems:
                return self._send(400, {"error": "no items"})
            if not JOB.start(kind, len(stems)):
                return self._send(409, {"error": "a job is already running"})
            threading.Thread(target=render_worker, args=(kind, stems, b), daemon=True).start()
            return self._send(200, {"ok": True})
        if u.path == "/api/select":
            it = PROJ.items.get(b.get("stem"))
            if not it:
                return self._send(404, {"error": "no such item"})
            it["seed"] = int(b.get("seed"))
            PROJ.save()
            return self._send(200, {"ok": True})
        if u.path == "/api/cancel":
            JOB.cancel = True
            with JOB.lock:
                if JOB.proc:
                    try:
                        JOB.proc.terminate()
                    except Exception:
                        pass
            LOG.put("cancel requested")
            return self._send(200, {"ok": True})
        if u.path == "/api/forget":
            PROJ.items.pop(b.get("stem"), None)
            PROJ.save()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "no route"})


UI = open(os.path.join(HERE, "jg4d_ui.html")).read() if \
    os.path.isfile(os.path.join(HERE, "jg4d_ui.html")) else "<h1>jg4d_ui.html missing</h1>"


def main():
    global UI
    p = os.path.join(HERE, "jg4d_ui.html")
    if os.path.isfile(p):
        UI = open(p).read()
    PROJ.load()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = "http://127.0.0.1:%d/" % PORT
    print("\n  jg4d vitrine  ->  %s\n  (ctrl-C to quit)\n" % url, flush=True)
    LOG.put("app started; project folder %s" % PROJ.out_dir)
    b = find_blender()
    LOG.put("blender: %s" % (b or "NOT FOUND - set JG4D_BLENDER"))
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
