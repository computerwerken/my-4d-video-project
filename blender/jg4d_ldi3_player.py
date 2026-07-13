# jg4d LDI3 player for Blender
# ----------------------------
# Imports Lifecast LDI3 volumetric photos/video (PNG/JPG sequences or stills)
# as three real displaced layer meshes, so Cycles/Eevee handle occlusion,
# lighting and stereo natively. Frame-perfect: geometry is rebuilt from the
# exact image of the current scene frame (image sequences, not movie codecs).
#
# Design notes (deliberately simple, for troubleshooting):
#   * Geometry: plain meshes displaced by a numpy decode of the depth cells,
#     driven by a frame_change handler. No Geometry Nodes, no drivers.
#   * Shading: ~8 shader nodes per layer (UV remap -> color cell -> emission,
#     alpha cell -> transparency). Two image datablocks per import: one sRGB
#     (color), one Non-Color (alpha; python reads depth from files directly).
#   * All LDI3 constants are editable in the N-panel and can be preloaded from
#     a JSON sidecar (inv_depth_coef / ftheta_scale / ftheta_inflation /
#     max_depth), so future VVE exports (DA3 / RAFT / FoundationStereo depth,
#     different scales) keep working without code changes.
#
# LDI3 layout reference: web/lifecast_res/LifecastVideoPlayerShaders11.js and
# lifecast_apps/source/ldi_common.cc in this repo. Grid: 3x3 cells;
# row 0 (bottom) = background layer, rows 1..2 = foreground layers;
# columns: color | inverse depth | alpha. 12-bit depth = lo/hi byte pair in
# the top half of the depth cell, with a fold/unfold error-correcting code.
#
# MIT License. LDI3 format and decode math (c) Lifecast Incorporated (MIT).

bl_info = {
    "name": "jg4d LDI3 player",
    "author": "jg + claude",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D View > Sidebar (N) > jg4d",
    "description": "Import Lifecast LDI3 volumetric photo/video as displaced layer meshes",
    "category": "Import-Export",
}

import bpy
import json
import os
import re
import numpy as np

NUM_LAYERS = 3

# Defaults matching the Lifecast web player. Override via sidecar JSON or panel.
DEFAULTS = {
    "inv_depth_coef": 0.3,     # s = inv_depth_coef / decoded_inverse_depth
    "ftheta_scale": 1.15,
    "ftheta_inflation": 3.0,
    "max_depth": 50.0,         # meters; also the clamp in the reference player
    "min_depth": 0.01,
    "decode_12bit": True,
    # 512 = geometric parity with the reference players (web/Unity/Unreal use
    # 512 quads per side). Color/alpha detail is independent of this: materials
    # sample the full-resolution texture per rendered pixel. Raise for even
    # finer depth silhouettes, lower for lighter scenes.
    "grid_n": 512,
}

# Caches. _red_cache: raw red-channel per file (one 5760^2 image feeds all
# three layers). _depth_cache: final per-vertex distances (tiny, keep many).
_red_cache = {}
_RED_CACHE_MAX = 4
_depth_cache = {}


# ----------------------------------------------------------------------------
# Decode helpers (pure numpy; mirrors LifecastVideoPlayerShaders11.js exactly)
# ----------------------------------------------------------------------------

def set_noncolor(img):
    """Mark an image as data (no color transform). OCIO configs vary, so try
    the common names in order; 'Linear' is the safe last resort."""
    for name in ("Non-Color", "Non-Colour Data", "Raw", "Linear"):
        try:
            img.colorspace_settings.name = name
            return
        except TypeError:
            continue


def _load_pixels_raw(filepath):
    """Load an image file and return HxWx1 float array of the RED channel,
    raw (no color management). Uses a temporary Non-Color datablock."""
    img = bpy.data.images.load(filepath, check_existing=False)
    try:
        set_noncolor(img)
        w, h = img.size
        buf = np.empty(w * h * img.channels, dtype=np.float32)
        img.pixels.foreach_get(buf)
        red = buf.reshape(h, w, img.channels)[:, :, 0]  # row 0 = image BOTTOM
        return np.ascontiguousarray(red)
    finally:
        bpy.data.images.remove(img)


def _sample_nearest(red, u, v):
    """Sample HxW array with normalized UV (v=0 is bottom row, like GL)."""
    h, w = red.shape
    x = np.clip((u * w).astype(np.int64), 0, w - 1)
    y = np.clip((v * h).astype(np.int64), 0, h - 1)
    return red[y, x]


def decode_inverse_depth(red, uv, layer, decode_12bit):
    """uv: (N,2) layer-local coordinates in [0,1]. Returns inverse depth (N,)."""
    u, v = uv[:, 0], uv[:, 1]
    cell_v = layer / 3.0  # depth cell offsets: x = 1/3, y = layer/3
    if decode_12bit:
        lo = _sample_nearest(red, 1 / 3 + u / 6.0, cell_v + (v + 1.0) / 6.0)
        hi = _sample_nearest(red, 1 / 3 + (u + 1.0) / 6.0, cell_v + (v + 1.0) / 6.0)
        lo = np.rint(lo * 255.0).astype(np.int64) & 255
        hi = np.rint(hi * 255.0).astype(np.int64) & 255
        hi //= 16
        lo = np.where(hi % 2 == 0, lo, 255 - lo)      # unfold ECC
        i12 = (lo & 255) | ((hi & 15) << 8)
        return np.clip(i12 / 4095.0, 1e-4, 1.0)
    else:
        val = _sample_nearest(red, 1 / 3 + u / 3.0, cell_v + v / 3.0)
        return np.clip(val, 1e-4, 1.0)


def compute_layer_scales(filepath, uv, layer, p):
    """Per-vertex radial distance s for one layer of one frame (cached)."""
    key = (os.path.abspath(filepath), layer, len(uv),
           p["inv_depth_coef"], p["max_depth"], p["min_depth"], p["decode_12bit"])
    if key in _depth_cache:
        return _depth_cache[key]
    apath = os.path.abspath(filepath)
    if apath not in _red_cache:
        if len(_red_cache) >= _RED_CACHE_MAX:
            _red_cache.pop(next(iter(_red_cache)))
        _red_cache[apath] = _load_pixels_raw(filepath)
    red = _red_cache[apath]
    invd = decode_inverse_depth(red, uv, layer, p["decode_12bit"])
    s = np.clip(p["inv_depth_coef"] / invd, p["min_depth"], p["max_depth"])
    _depth_cache[key] = s.astype(np.float32)
    return _depth_cache[key]


# ----------------------------------------------------------------------------
# Mesh construction (mirrors Ldi3Mesh.js makeEquiangularMesh)
# ----------------------------------------------------------------------------

def grid_dirs_uvs(grid_n, ftheta_scale, inflation):
    """Pure numpy: unit ray directions + layer-local UVs for the vertex grid.
    Safe to call anywhere (no bpy). Used by mesh build AND the frame handler."""
    n = grid_n
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))  # (n+1, n+1)
    u = (i / n).ravel()
    v = (j / n).ravel()
    a = 2.0 * (u - 0.5)
    b = 2.0 * (v - 0.5)
    theta = np.arctan2(b, a)
    r = np.sqrt(a * a + b * b) / ftheta_scale
    r = 0.5 * r + 0.5 * np.power(r, inflation)
    phi = r * np.pi / 2.0
    # Lifecast frame: +x right, +y up, -z forward (dir = (cos t sin p,
    # sin t sin p, -cos p)). Blender is z-up; we put capture-forward on +Y:
    # blender (x, y, z) = (lc_x, -lc_z, lc_y)
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.cos(phi),
                     np.sin(theta) * np.sin(phi)], axis=1).astype(np.float32)
    uvs = np.stack([u, v], axis=1).astype(np.float32)
    return dirs, uvs


def build_layer_mesh(name, grid_n, ftheta_scale, inflation):
    """Unit-radius equiangular dome mesh with circular clip.
    Returns (object, unit_directions (N,3), uvs (N,2))."""
    n = grid_n
    dirs, uvs = grid_dirs_uvs(n, ftheta_scale, inflation)

    # quads inside the image circle only (same clip rule as the JS player)
    margin = 2
    qi, qj = np.meshgrid(np.arange(n), np.arange(n))
    di = qi - n / 2.0
    dj = qj - n / 2.0
    keep = (di * di + dj * dj) <= ((n + margin) ** 2) / 4.0
    qi, qj = qi[keep], qj[keep]
    a0 = qi + (n + 1) * qj
    faces = np.stack([a0, a0 + (n + 1), a0 + (n + 1) + 1, a0 + 1], axis=1)  # quads

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(dirs.tolist(), [], faces.tolist())
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_verts = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    uv_layer.data.foreach_set("uv", uvs[loop_verts].ravel())
    mesh.validate()
    obj = bpy.data.objects.new(name, mesh)
    return obj, dirs.astype(np.float32), uvs


def displace_layer(obj, dirs, s):
    """Set vertex positions = unit_dir * s."""
    co = (dirs * s[:, None]).astype(np.float32)
    obj.data.vertices.foreach_set("co", co.ravel())
    obj.data.update()


# ----------------------------------------------------------------------------
# Materials: color cell -> emission, alpha cell -> transparency
# ----------------------------------------------------------------------------

def make_layer_material(name, img_color, img_alpha, layer, seq_settings):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "HASHED"      # order-independent, writes depth (Eevee)
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "HASHED"
    nt = mat.node_tree
    nt.nodes.clear()

    def node(t, x, y):
        n = nt.nodes.new(t)
        n.location = (x, y)
        return n

    uvmap = node("ShaderNodeUVMap", -900, 0)
    uvmap.uv_map = "UVMap"

    # layer-local UV -> grid cell UV:  cell = uv/3 + (cell_x/3, layer/3)
    map_color = node("ShaderNodeMapping", -700, 100)
    map_color.inputs["Scale"].default_value = (1 / 3, 1 / 3, 1)
    map_color.inputs["Location"].default_value = (0.0, layer / 3.0, 0)
    map_alpha = node("ShaderNodeMapping", -700, -250)
    map_alpha.inputs["Scale"].default_value = (1 / 3, 1 / 3, 1)
    map_alpha.inputs["Location"].default_value = (2 / 3, layer / 3.0, 0)

    tex_color = node("ShaderNodeTexImage", -450, 100)
    tex_color.image = img_color
    tex_alpha = node("ShaderNodeTexImage", -450, -250)
    tex_alpha.image = img_alpha
    for tex in (tex_color, tex_alpha):
        tex.extension = "EXTEND"
        if seq_settings:
            tex.image_user.frame_start = seq_settings["frame_start"]
            tex.image_user.frame_offset = seq_settings["frame_offset"]
            tex.image_user.frame_duration = seq_settings["frame_duration"]
            tex.image_user.use_auto_refresh = True
            tex.image_user.use_cyclic = True

    emit = node("ShaderNodeEmission", -150, 100)
    transp = node("ShaderNodeBsdfTransparent", -150, -100)
    mixer = node("ShaderNodeMixShader", 50, 0)
    out = node("ShaderNodeOutputMaterial", 250, 0)

    nt.links.new(uvmap.outputs["UV"], map_color.inputs["Vector"])
    nt.links.new(uvmap.outputs["UV"], map_alpha.inputs["Vector"])
    nt.links.new(map_color.outputs["Vector"], tex_color.inputs["Vector"])
    nt.links.new(map_alpha.outputs["Vector"], tex_alpha.inputs["Vector"])
    nt.links.new(tex_color.outputs["Color"], emit.inputs["Color"])
    nt.links.new(tex_alpha.outputs["Color"], mixer.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mixer.inputs[1])   # Fac=0 -> transparent
    nt.links.new(emit.outputs["Emission"], mixer.inputs[2]) # Fac=1 -> footage
    nt.links.new(mixer.outputs["Shader"], out.inputs["Surface"])
    return mat


# ----------------------------------------------------------------------------
# Sequence bookkeeping
# ----------------------------------------------------------------------------

def parse_sequence(filepath):
    """'/x/uno_ldi3_000030.png' -> (pattern '/x/uno_ldi3_%06d.png', 30) or None."""
    d, base = os.path.split(filepath)
    m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", base)
    if not m:
        return None
    stem, digits, ext = m.groups()
    pattern = os.path.join(d, "%s%%0%dd%s" % (stem, len(digits), ext))
    frames = sorted(
        int(re.match(r"^%s(\d+)%s$" % (re.escape(stem), re.escape(ext)), f).group(1))
        for f in os.listdir(d)
        if re.match(r"^%s(\d+)%s$" % (re.escape(stem), re.escape(ext)), f))
    if len(frames) < 2:
        return None
    return {"pattern": pattern, "first": frames[0], "count": len(frames)}


def load_sidecar(filepath):
    """Look for <file>.json or jg4d_sidecar.json next to the media."""
    for cand in (os.path.splitext(filepath)[0] + ".json",
                 os.path.join(os.path.dirname(filepath), "jg4d_sidecar.json")):
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    data = json.load(f)
                return {k: data[k] for k in DEFAULTS if k in data}
            except Exception as e:
                print("jg4d: failed to read sidecar %s: %s" % (cand, e))
    return {}


# ----------------------------------------------------------------------------
# The import operator
# ----------------------------------------------------------------------------

def rig_props(root):
    return json.loads(root["jg4d"])


def import_ldi3(context, filepath):
    p = dict(DEFAULTS)
    p.update(load_sidecar(filepath))

    seq = parse_sequence(filepath)

    root = bpy.data.objects.new("jg4d_ldi3", None)
    root.empty_display_size = 0.2
    context.scene.collection.objects.link(root)

    # image datablocks (color=sRGB for shading, alpha=Non-Color for exact values)
    img_color = bpy.data.images.load(filepath, check_existing=False)
    img_alpha = bpy.data.images.load(filepath, check_existing=False)
    set_noncolor(img_alpha)
    seq_settings = None
    if seq:
        for img in (img_color, img_alpha):
            img.source = "SEQUENCE"
        seq_settings = {
            # with frame_start=1 and offset=first-1, scene frame N shows file N
            "frame_start": 1,
            "frame_offset": seq["first"] - 1,
            "frame_duration": seq["count"],
        }

    for layer in range(NUM_LAYERS):
        name = "jg4d_layer%d" % layer
        obj, dirs, uvs = build_layer_mesh(name, p["grid_n"],
                                          p["ftheta_scale"], p["ftheta_inflation"])
        context.scene.collection.objects.link(obj)
        obj.parent = root
        obj.data.materials.append(
            make_layer_material(name, img_color, img_alpha, layer, seq_settings))
        s = compute_layer_scales(filepath, uvs, layer, p)
        displace_layer(obj, dirs, s)

    root["jg4d"] = json.dumps({
        "params": p,
        "filepath": filepath,
        "pattern": seq["pattern"] if seq else "",
        "first": seq["first"] if seq else 0,
        "count": seq["count"] if seq else 1,
    })
    return root


def refresh_rig(root, scene):
    """Re-decode depth for the current frame and re-displace all layers."""
    cfg = rig_props(root)
    p = cfg["params"]
    if cfg["pattern"]:
        f = scene.frame_current
        f = cfg["first"] + (f - 1) % cfg["count"]  # cyclic, frame 1 = first file
        path = cfg["pattern"] % f
    else:
        path = cfg["filepath"]
    if not os.path.isfile(path):
        return
    for child in root.children:
        m = re.match(r"^jg4d_layer(\d)", child.name)
        if not m:
            continue
        layer = int(m.group(1))
        nverts = len(child.data.vertices)
        # unit directions from mesh: recompute from stored s? Simpler: rebuild
        # dirs from UV attribute (cheap, cached alongside the object).
        dirs, uvs = _rig_geom_cache_get(child, p)
        s = compute_layer_scales(path, uvs, layer, p)
        if len(s) == nverts:
            displace_layer(child, dirs, s)


_geom_cache = {}

def _rig_geom_cache_get(obj, p):
    """Pure-numpy grid regen (NO bpy object churn: this runs inside the
    frame_change handler, where creating/removing IDs crashes Blender)."""
    key = (p["grid_n"], p["ftheta_scale"], p["ftheta_inflation"])
    if key not in _geom_cache:
        _geom_cache[key] = grid_dirs_uvs(*key)
    return _geom_cache[key]


def jg4d_frame_handler(scene, _depsgraph=None):
    for obj in scene.collection.all_objects:
        if obj.type == "EMPTY" and "jg4d" in obj:
            if json.loads(obj["jg4d"])["pattern"]:
                refresh_rig(obj, scene)


# ----------------------------------------------------------------------------
# Dubois red-cyan anaglyph conversion (identical matrices + gamma handling to
# the jg4dplayer web app's composite shader, so previews match the player).
# Use for PREVIEWS while animating; for delivery keep L/R masters separate and
# apply Dubois after the grade.
# ----------------------------------------------------------------------------

DUBOIS_L = np.array([[ 0.4561000,  0.5004840,  0.1763810],
                     [-0.0400822, -0.0378246, -0.0157589],
                     [-0.0152161, -0.0205971, -0.0054686]], np.float32)
DUBOIS_R = np.array([[-0.0434706, -0.0879388, -0.0015553],
                     [ 0.3784760,  0.7336400, -0.0180517],
                     [-0.0721527, -0.1129610,  1.2264000]], np.float32)


def dubois_pair(left_srgb, right_srgb):
    """(H,W,3) float sRGB-encoded in [0,1] -> Dubois red-cyan, same encoding."""
    l = np.power(np.clip(left_srgb, 0, 1), 2.2)
    r = np.power(np.clip(right_srgb, 0, 1), 2.2)
    out = l @ DUBOIS_L.T + r @ DUBOIS_R.T
    return np.power(np.clip(out, 0, 1), 1.0 / 2.2)


def _read_rgb(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        set_noncolor(img)  # already display-encoded pixels; read them raw
        w, h = img.size
        buf = np.empty(w * h * img.channels, np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(h, w, img.channels)[:, :, :3][::-1].copy()  # top-down
    finally:
        bpy.data.images.remove(img)


def _write_rgb(path, rgb_topdown):
    h, w = rgb_topdown.shape[:2]
    img = bpy.data.images.new("__jg4d_out", w, h, alpha=False)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[:, :, :3] = rgb_topdown[::-1]
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def dubois_convert_files(left_path, right_path, out_path):
    _write_rgb(out_path, dubois_pair(_read_rgb(left_path), _read_rgb(right_path)))


# ----------------------------------------------------------------------------
# Operators + panel
# ----------------------------------------------------------------------------

class JG4D_OT_import(bpy.types.Operator):
    bl_idname = "jg4d.import_ldi3"
    bl_label = "Import LDI3 (photo / PNG-JPG sequence)"
    bl_options = {"REGISTER", "UNDO"}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg", options={"HIDDEN"})

    def execute(self, context):
        try:
            root = import_ldi3(context, self.filepath)
        except Exception as e:
            self.report({"ERROR"}, "jg4d import failed: %s" % e)
            return {"CANCELLED"}
        cfg = rig_props(root)
        self.report({"INFO"}, "Imported LDI3 (%s, %d frame(s))" %
                    ("sequence" if cfg["pattern"] else "still", cfg["count"]))
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class JG4D_OT_refresh(bpy.types.Operator):
    """Re-decode depth for the current frame (after changing parameters)"""
    bl_idname = "jg4d.refresh"
    bl_label = "Refresh depth"

    def execute(self, context):
        _depth_cache.clear()
        for obj in context.scene.collection.all_objects:
            if obj.type == "EMPTY" and "jg4d" in obj:
                refresh_rig(obj, context.scene)
        return {"FINISHED"}


class JG4D_OT_stereo(bpy.types.Operator):
    """Enable stereoscopic rendering + red-cyan anaglyph preview at the capture point"""
    bl_idname = "jg4d.setup_stereo"
    bl_label = "Setup stereo camera (anaglyph)"

    def execute(self, context):
        scene = context.scene
        scene.render.use_multiview = True
        scene.render.views_format = "STEREO_3D"
        scene.render.image_settings.views_format = "STEREO_3D"
        scene.render.image_settings.stereo_3d_format.display_mode = "ANAGLYPH"
        scene.render.image_settings.stereo_3d_format.anaglyph_type = "RED_CYAN"
        cam = scene.camera
        if cam is None:
            cam_data = bpy.data.cameras.new("jg4d_cam")
            cam = bpy.data.objects.new("jg4d_cam", cam_data)
            scene.collection.objects.link(cam)
            scene.camera = cam
        cam.location = (0, 0, 0)
        cam.rotation_euler = (1.5708, 0, 0)  # look down -Y (lifecast forward)
        cam.data.stereo.interocular_distance = 0.063
        cam.data.stereo.convergence_mode = "OFFAXIS"
        cam.data.stereo.convergence_distance = 2.0
        self.report({"INFO"}, "Stereo enabled. Viewport: View > Stereoscopy.")
        return {"FINISHED"}


class JG4D_OT_dubois(bpy.types.Operator):
    """Convert an L/R pair (Blender's *_L/*_R suffixed renders) to a Dubois
    red-cyan anaglyph PNG next to them. Select the LEFT file."""
    bl_idname = "jg4d.dubois"
    bl_label = "Dubois anaglyph from L/R pair"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.exr;*.tif;*.tiff", options={"HIDDEN"})

    def execute(self, context):
        lp = self.filepath
        for l_tag, r_tag in (("_L", "_R"), ("left", "right"), ("_l", "_r")):
            if l_tag in os.path.basename(lp):
                rp = os.path.join(os.path.dirname(lp),
                                  os.path.basename(lp).replace(l_tag, r_tag))
                if os.path.isfile(rp):
                    break
        else:
            self.report({"ERROR"}, "Could not find matching right-eye file")
            return {"CANCELLED"}
        out = os.path.splitext(lp)[0].replace(l_tag, "") + "_anaglyph.png"
        try:
            dubois_convert_files(lp, rp, out)
        except Exception as e:
            self.report({"ERROR"}, "Dubois conversion failed: %s" % e)
            return {"CANCELLED"}
        self.report({"INFO"}, "Wrote %s" % out)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class JG4D_PT_panel(bpy.types.Panel):
    bl_label = "jg4d LDI3 player"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "jg4d"

    def draw(self, context):
        col = self.layout.column()
        col.operator("jg4d.import_ldi3", icon="IMPORT")
        col.operator("jg4d.refresh", icon="FILE_REFRESH")
        col.operator("jg4d.setup_stereo", icon="CAMERA_STEREO")
        col.operator("jg4d.dubois", icon="IMAGE_RGB")
        col.separator()
        col.label(text="Params live in the sidecar JSON or")
        col.label(text="the 'jg4d' prop on the import root.")
        col.label(text="Convert mp4 to frames with:")
        col.label(text="ffmpeg -i in.mp4 out_%06d.png")


CLASSES = (JG4D_OT_import, JG4D_OT_refresh, JG4D_OT_stereo, JG4D_OT_dubois, JG4D_PT_panel)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    if jg4d_frame_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(jg4d_frame_handler)


def unregister():
    if jg4d_frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(jg4d_frame_handler)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
