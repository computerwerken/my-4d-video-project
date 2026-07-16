# jg4d vitrine - Blender side
# ---------------------------
# Runs INSIDE Blender, headless. Not an addon; the app shells out to:
#
#   blender --background --factory-startup --python jg4d_blender_job.py -- <json>
#
# --factory-startup on purpose: whatever is in your normal startup file (an addon
# mid-upgrade, a stray unit setting, a leftover scene) has no business deciding
# what a batch render looks like. This gives every job the same empty room.
#
# The <json> is a job spec:
#   {"kit_dir": "...", "albedo": "...", "out": "...", "mode": "contact"|"final",
#    "look": {...}, "seeds": [1,2,3,4], "samples": 48, "res_x": 4096, "res_y": 2160,
#    "res_pct": 25}
#
# contact -> one anaglyph PNG per seed, cheap, for picking a variant
# final   -> Individual L/R masters at full quality, for the Resolve grade
#
# Everything it prints with a "JG4D:" prefix is parsed by the app. Everything else
# is Blender's own noise and gets shown in the log verbatim.
#
# MIT License.

import json
import os
import sys
import traceback


def log(msg):
    print("JG4D: %s" % msg, flush=True)


def fail(msg):
    print("JG4D-ERROR: %s" % msg, flush=True)
    sys.exit(1)


def main():
    if "--" not in sys.argv:
        fail("no job spec (expected: --python this.py -- '<json>')")
    raw = sys.argv[sys.argv.index("--") + 1]
    if os.path.isfile(raw):
        with open(raw) as f:
            job = json.load(f)
    else:
        job = json.loads(raw)

    import bpy

    kit_dir = job["kit_dir"]
    if kit_dir not in sys.path:
        sys.path.insert(0, kit_dir)

    try:
        import vitrine_kit
    except Exception as e:
        fail("cannot import vitrine_kit from %s: %s" % (kit_dir, e))

    # Register directly rather than installing an addon: a batch job should not
    # mutate the user's Blender config, and --factory-startup means the addon
    # would not be enabled anyway.
    try:
        vitrine_kit.register()
    except Exception as e:
        fail("vitrine_kit.register() failed: %s" % e)
    log("vitrine_kit %s registered" % (".".join(str(v) for v in vitrine_kit.bl_info["version"])))

    scene = bpy.context.scene
    props = scene.jg4d_vitrine

    albedo = job["albedo"]
    if not os.path.isfile(albedo):
        fail("albedo not found: %s" % albedo)
    props.albedo_path = albedo

    # --- resolution BEFORE build: safe ruling reads resolution_y ---------------
    scene.render.resolution_x = int(job.get("res_x", 4096))
    scene.render.resolution_y = int(job.get("res_y", 2160))
    scene.render.resolution_percentage = int(job.get("res_pct", 100))

    look = job.get("look", {})
    for k, v in look.items():
        if hasattr(props, k):
            try:
                setattr(props, k, v)
            except Exception as e:
                log("look: could not set %s=%r (%s)" % (k, v, e))
        else:
            log("look: unknown property %s, ignored" % k)

    scene.render.engine = "CYCLES"
    try:
        scene.cycles.samples = int(job.get("samples", 256))
        scene.cycles.use_denoising = bool(job.get("denoise", True))
        scene.cycles.volume_bounces = max(2, scene.cycles.volume_bounces)
        scene.cycles.transmission_bounces = max(8, scene.cycles.transmission_bounces)
    except Exception as e:
        log("cycles settings partly unavailable: %s" % e)

    device = job.get("device", "GPU")
    if device == "GPU":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
                try:
                    prefs.compute_device_type = backend
                except TypeError:
                    continue
                prefs.get_devices()
                usable = [d for d in prefs.devices if d.type != "CPU"]
                if usable:
                    for d in prefs.devices:
                        d.use = True
                    scene.cycles.device = "GPU"
                    log("GPU: %s (%s)" % (backend, ", ".join(d.name for d in usable)))
                    break
            else:
                log("no GPU backend available, using CPU")
        except Exception as e:
            log("GPU setup failed (%s); using CPU" % e)

    mode = job.get("mode", "contact")
    seeds = job.get("seeds") or [1]
    out_dir = job["out"]
    os.makedirs(out_dir, exist_ok=True)
    stem = job.get("stem") or os.path.basename(albedo).replace("_albedo.png", "")

    results = []
    for seed in seeds:
        props.seed_curl = int(seed)
        props.seed_scuff = int(seed)
        props.seed_dust = int(seed)
        props.seed_light = int(seed)

        r = bpy.ops.jg4d.vitrine_build()
        if "FINISHED" not in r:
            fail("vitrine_build returned %s (seed %d)" % (r, seed))

        # Order matters and it is not cosmetic: safe ruling needs the camera to
        # exist and the resolution to be set, and solve stereo needs the built
        # case geometry. Both are recomputed per seed because a rebuild can move
        # things. Two cheap calls; skipping them is how you get aliased dots and
        # a 7x parallax budget.
        bpy.ops.jg4d.vitrine_safe_ruling()
        bpy.ops.jg4d.vitrine_solve_stereo()

        cam = scene.camera
        log("seed %d: ruling %.1f lpi, interaxial %.2f mm, converge %.1f mm"
            % (seed, props.halftone_ruling,
               cam.data.stereo.interocular_distance * 1000.0,
               cam.data.stereo.convergence_distance * 1000.0))

        if mode == "contact":
            # One anaglyph frame per seed: you want to SEE the 3D when picking,
            # not judge depth from a flat left eye.
            scene.render.image_settings.views_format = "STEREO_3D"
            scene.render.image_settings.stereo_3d_format.display_mode = "ANAGLYPH"
            scene.render.image_settings.stereo_3d_format.anaglyph_type = "RED_CYAN"
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_depth = "8"
            path = os.path.join(out_dir, "%s_seed%03d.png" % (stem, seed))
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            results.append({"seed": seed, "path": path})
            log("wrote %s" % path)
        else:
            # Individual L/R, 16-bit, straight into the Resolve grade. NOT
            # anaglyph: Blender's is a plain matrix, not Dubois, and baking any
            # anaglyph before the grade collapses the image you meant to grade.
            scene.render.image_settings.views_format = "INDIVIDUAL"
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.image_settings.color_depth = "16"
            base = os.path.join(out_dir, "%s_" % stem)
            scene.render.filepath = base
            bpy.ops.render.render(write_still=True)
            made = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
                    if f.startswith(stem + "_") and ("_L" in f or "_R" in f)]
            results.append({"seed": seed, "paths": made})
            log("wrote %d view(s) for %s" % (len(made), stem))

    print("JG4D-RESULT: %s" % json.dumps(results), flush=True)
    log("done")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("JG4D-ERROR: %s" % traceback.format_exc().strip().splitlines()[-1], flush=True)
        sys.exit(1)
