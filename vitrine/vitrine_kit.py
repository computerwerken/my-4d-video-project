# jg4d vitrine kit for Blender
# ----------------------------
# Builds a museum-case rig around a scanned archival item: a real sheet of paper
# with curl and edge thickness, a scuffed glass pane, a dust-bearing volume, and
# case lighting -- sitting in black, aimed at a red-cyan anaglyph delivery.
#
# Companion to vitrine_prep.py, which turns a scan into the four maps this reads
# (albedo / alpha / ink / scuff) plus a JSON sidecar with the item's real size.
# Pick any *_albedo.png; the siblings and sidecar are found automatically.
#
# Design notes, same house style as jg4d_ldi3_player.py:
#   * Geometry is plain meshes built with numpy. No Geometry Nodes, no drivers.
#     Dust drifts via a frame_change_post handler, the same mechanism the LDI3
#     player already uses to re-displace its layers.
#   * Everything random is derived from an explicit integer seed, so a look is
#     reproducible and the Randomize buttons are just "seed = seed + 1, rebuild".
#   * Nothing here needs the LDI3 player. It coexists with it (same N-panel tab),
#     and because both are ordinary Cycles scenes you can put a vitrine INSIDE a
#     captured LDI3 scene later if the film wants that.
#
# Three things this file is opinionated about, because they are where a vitrine
# shot in anaglyph actually goes wrong:
#
#   1. INTERAXIAL. The LDI3 player's 63 mm is human eye spacing, correct for a
#      captured room. A vitrine is a small subject half a metre away; 63 mm there
#      is hyperstereo and gives you a doll's house with unfusable edges. Solve
#      stereo computes the interaxial from the case geometry and a target
#      on-screen parallax instead. Expect ~10-20 mm.
#
#   2. CONVERGENCE. Put it exactly on the glass. The theatre screen then IS the
#      vitrine pane: the item sits in positive parallax behind it, the scuffs
#      land at zero parallax where they are sharpest and most legible, and no
#      window violation is possible because nothing is in front of the glass.
#
#   3. HALFTONE RULING. Resynthesising the scan's true 133 lpi is a trap: a
#      280 mm sheet filling a 2160 px frame gives ~1.5 px per halftone cell, so
#      you would rebuild the exact aliasing you just removed. Safe ruling
#      computes what the delivery frame can actually carry (~3 px/cell) and
#      defaults to it. The dots then read as ink, not as sampling error.
#
# MIT License.

bl_info = {
    "name": "jg4d vitrine kit",
    "author": "jg + claude",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "3D View > Sidebar (N) > jg4d",
    "description": "Museum-vitrine rig for scanned archival items, built for anaglyph delivery",
    "category": "Object",
}

import bpy
import json
import math
import os

import numpy as np

MM = 0.001  # Blender is metres; every dimension in this file is authored in mm.

TAG = "jg4d_vitrine"


# ----------------------------------------------------------------------------
# Compatibility helpers
#
# Principled BSDF socket names moved in 4.0 (Transmission -> Transmission Weight
# and friends), and blend_method disappeared in 4.2's EEVEE Next. Same defensive
# approach as set_noncolor() in the LDI3 player: try the known names in order.
# ----------------------------------------------------------------------------

def set_input(node, names, value):
    """Set the first socket that exists from `names`. Returns True if set."""
    if isinstance(names, str):
        names = (names,)
    for n in names:
        if n in node.inputs:
            try:
                node.inputs[n].default_value = value
                return True
            except (TypeError, ValueError):
                pass
    return False


def get_input(node, names):
    if isinstance(names, str):
        names = (names,)
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def set_noncolor(img):
    for name in ("Non-Color", "Non-Colour Data", "Raw", "Linear"):
        try:
            img.colorspace_settings.name = name
            return
        except TypeError:
            continue


def set_blend(mat, mode="BLEND"):
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = mode
        except TypeError:
            pass


def load_image(path, noncolor=False):
    if not path or not os.path.isfile(path):
        return None
    img = bpy.data.images.load(path, check_existing=True)
    if noncolor:
        set_noncolor(img)
    return img


# ----------------------------------------------------------------------------
# Sidecar
# ----------------------------------------------------------------------------

DEFAULTS = {
    "physical_mm": [180.0, 240.0],
    "paper_rgb": [0.85, 0.80, 0.70],
    "resynth_halftone": {"ruling_lpi": 100.0, "angle_deg": 45.0},
}


def item_paths(albedo_path):
    """<stem>_albedo.png -> the sibling maps and the sidecar."""
    d = os.path.dirname(albedo_path)
    base = os.path.basename(albedo_path)
    stem = base
    for suf in ("_albedo.png", "_albedo.jpg", "_albedo.tif", "_albedo.exr"):
        if base.lower().endswith(suf):
            stem = base[:-len(suf)]
            break
    else:
        stem = os.path.splitext(base)[0]
    ext = os.path.splitext(base)[1]

    def sib(kind):
        p = os.path.join(d, stem + "_" + kind + ext)
        return p if os.path.isfile(p) else None

    return {
        "stem": stem,
        "albedo": albedo_path,
        "alpha": sib("alpha"),
        "ink": sib("ink"),
        "scuff": sib("scuff"),
        "json": os.path.join(d, stem + ".json"),
    }


def load_sidecar(albedo_path):
    p = item_paths(albedo_path)
    cfg = json.loads(json.dumps(DEFAULTS))
    if os.path.isfile(p["json"]):
        try:
            with open(p["json"]) as f:
                data = json.load(f)
            for k in ("physical_mm", "paper_rgb", "resynth_halftone", "screen",
                      "ppi", "pixels", "matte_mode"):
                if k in data and data[k] is not None:
                    cfg[k] = data[k]
        except Exception as e:
            print("jg4d vitrine: could not read sidecar %s: %s" % (p["json"], e))
    cfg["paths"] = p
    return cfg


# ----------------------------------------------------------------------------
# Stereo math
#
# Parallax of a point at depth Z, converged at Zc, as a fraction of frame width:
#     p = I * (1/Zc - 1/Z) / (2 * tan(hfov/2))
# Positive = behind the screen. Solve it for I against a target and you get an
# interaxial that is correct for THIS case at THIS lens, which is the only way
# to keep a small subject fusable.
# ----------------------------------------------------------------------------

def sensor_extents(cam_data, rx, ry):
    """(horizontal, vertical) sensor extent in mm for this fit mode.

    AUTO does not mean 'use sensor_height for the vertical' -- it means
    sensor_width maps to the LARGER pixel dimension and the other is derived
    from the aspect. Getting this wrong quietly scales every parallax number,
    so it is worth the eight lines.
    """
    sw, sh = cam_data.sensor_width, cam_data.sensor_height
    fit = cam_data.sensor_fit
    aspect = rx / float(max(ry, 1))
    if fit == "VERTICAL":
        return sh * aspect, sh
    if fit == "HORIZONTAL":
        return sw, sw / max(aspect, 1e-9)
    if rx >= ry:                       # AUTO, landscape
        return sw, sw / max(aspect, 1e-9)
    return sw * aspect, sw             # AUTO, portrait


def hfov_of(cam_obj, scene):
    h, _ = sensor_extents(cam_obj.data, scene.render.resolution_x,
                          scene.render.resolution_y)
    return 2.0 * math.atan(h / (2.0 * cam_obj.data.lens))


def vfov_of(cam_obj, scene):
    _, v = sensor_extents(cam_obj.data, scene.render.resolution_x,
                          scene.render.resolution_y)
    return 2.0 * math.atan(v / (2.0 * cam_obj.data.lens))


def parallax_frac(I, Zc, Z, hfov):
    if Z <= 1e-6 or Zc <= 1e-6:
        return 0.0
    return I * (1.0 / Zc - 1.0 / Z) / (2.0 * math.tan(hfov / 2.0))


def solve_interaxial(Zc, Z_far, hfov, target_frac):
    """Largest interaxial whose far plane still lands inside target_frac."""
    denom = (1.0 / Zc - 1.0 / Z_far)
    if abs(denom) < 1e-9:
        return 0.065
    return abs(2.0 * target_frac * math.tan(hfov / 2.0) / denom)


def safe_ruling(item_mm, on_screen_px, px_per_cell=3.0):
    """Highest halftone ruling the delivery frame can actually carry.

    item_mm is the item's height in mm, on_screen_px how tall it will sit in the
    frame. Below ~3 px per cell a screen stops being dots and becomes shimmer --
    and shimmer is per-eye noise, which in red-cyan is the one thing that will
    not fuse.
    """
    if item_mm <= 0 or on_screen_px <= 0:
        return 100.0
    ppi_on_screen = on_screen_px / (item_mm / 25.4)
    return max(10.0, ppi_on_screen / px_per_cell)


# ----------------------------------------------------------------------------
# Paper mesh
#
# A sheet that has lived in a book is not a plane and it is not noise: it is a
# developable surface. Mostly a cylindrical bow from the spine, plus corner lift
# where the fibres have relaxed, plus a slow ripple. Dead-flat paper reads as a
# billboard in stereo; this reads as an artifact, and it is what makes the
# raking light do anything at all.
# ----------------------------------------------------------------------------

def paper_curl(u, v, seed, amount_mm, aspect):
    rng = np.random.default_rng(int(seed))
    a0 = rng.uniform(0.0, math.pi)
    cu, cv = (u - 0.5) * 2.0, (v - 0.5) * 2.0 * aspect
    a = cu * math.cos(a0) - cv * math.sin(a0)
    b = cu * math.sin(a0) + cv * math.cos(a0)

    bow = rng.uniform(0.35, 1.0) * rng.choice([-1.0, 1.0])
    dish = rng.uniform(0.2, 0.9)
    twist = rng.uniform(-0.35, 0.35)

    z = bow * (a ** 2) * 0.5
    z += dish * (a ** 2) * (b ** 2) * 0.35          # corners lift
    z += twist * a * b * 0.3

    for _ in range(int(rng.integers(2, 4))):        # slow ripple
        f = rng.uniform(0.6, 2.2)
        ph = rng.uniform(0, 2 * math.pi)
        ax = rng.uniform(0, math.pi)
        d = a * math.cos(ax) + b * math.sin(ax)
        z += math.sin(f) * 0.06 * np.sin(d * f * math.pi + ph)

    z -= z.mean()
    peak = max(np.abs(z).max(), 1e-9)
    return (z / peak) * (amount_mm * MM)


def build_paper(name, w_mm, h_mm, seed, curl_mm, res=96):
    n = int(max(8, res))
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    u = (i / n).ravel().astype(np.float64)
    v = (j / n).ravel().astype(np.float64)

    aspect = h_mm / max(w_mm, 1e-6)
    z = paper_curl(u, v, seed, curl_mm, aspect)

    x = (u - 0.5) * w_mm * MM
    y = (v - 0.5) * h_mm * MM
    verts = np.stack([x, y, z], axis=1)

    qi, qj = np.meshgrid(np.arange(n), np.arange(n))
    a0 = (qi + (n + 1) * qj).ravel()
    faces = np.stack([a0, a0 + 1, a0 + (n + 1) + 1, a0 + (n + 1)], axis=1)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    uvl = mesh.uv_layers.new(name="UVMap")
    loop_verts = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    uvs = np.stack([u, v], axis=1)
    uvl.data.foreach_set("uv", uvs[loop_verts].ravel())
    mesh.validate()
    if hasattr(mesh, "shade_smooth"):
        mesh.shade_smooth()
    else:
        for p in mesh.polygons:
            p.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    sol = obj.modifiers.new("thickness", "SOLIDIFY")
    sol.thickness = 0.22 * MM        # ~0.22 mm: a sheet of book stock
    sol.offset = 0.0
    sol.use_even_offset = True
    return obj


def reshape_paper(obj, w_mm, h_mm, seed, curl_mm):
    """Re-displace an existing paper mesh in place (Randomize curl)."""
    me = obj.data
    n_v = len(me.vertices)
    n = int(round(math.sqrt(n_v))) - 1
    if (n + 1) ** 2 != n_v:
        return False
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    u = (i / n).ravel().astype(np.float64)
    v = (j / n).ravel().astype(np.float64)
    z = paper_curl(u, v, seed, curl_mm, h_mm / max(w_mm, 1e-6))
    co = np.stack([(u - 0.5) * w_mm * MM, (v - 0.5) * h_mm * MM, z], axis=1)
    me.vertices.foreach_set("co", co.ravel().astype(np.float32))
    me.update()
    return True


# ----------------------------------------------------------------------------
# Simple primitives (numpy, so no operator context needed)
# ----------------------------------------------------------------------------

def build_box(name, w, h, d, center=(0, 0, 0)):
    x, y, z = w / 2.0, d / 2.0, h / 2.0
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], np.float64)
    v += np.array(center)
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
         [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f)
    me.validate()
    return bpy.data.objects.new(name, me)


def box_uvs(me, w, h):
    """Project every face on XZ into 0..1 over the box's own width/height.

    The front face comes out correct, the edge-on faces come out degenerate,
    and since they are 5 mm of glass seen edge-on that is exactly what we want.
    Gives the painted scuff-map override a sane place to land.
    """
    uvl = me.uv_layers.new(name="UVMap") if not me.uv_layers else me.uv_layers[0]
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lv = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lv)
    p = co[lv]
    u = p[:, 0] / max(w, 1e-9) + 0.5
    v = p[:, 2] / max(h, 1e-9) + 0.5
    uvl.data.foreach_set("uv", np.stack([u, v], 1).ravel().astype(np.float32))


def build_plane(name, w, h, center=(0, 0, 0), axis="XZ"):
    x, z = w / 2.0, h / 2.0
    if axis == "XZ":
        v = np.array([[-x, 0, -z], [x, 0, -z], [x, 0, z], [-x, 0, z]], np.float64)
    else:
        v = np.array([[-x, -z, 0], [x, -z, 0], [x, z, 0], [-x, z, 0]], np.float64)
    v += np.array(center)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], [[0, 1, 2, 3]])
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    me.validate()
    return bpy.data.objects.new(name, me)


def build_dust(name, count, w, h, d, seed, center=(0, 0, 0), size_mm=0.35):
    """One mesh, `count` tiny tetrahedra. Tetrahedra rather than icospheres
    because at 0.3 mm they are 1-2 px and nobody will ever count the faces, but
    4 tris x 400 motes stays cheap and, unlike a point cloud, they catch light."""
    rng = np.random.default_rng(int(seed))
    p = np.stack([rng.uniform(-w / 2, w / 2, count),
                  rng.uniform(-d / 2, d / 2, count),
                  rng.uniform(-h / 2, h / 2, count)], axis=1) + np.array(center)
    s = size_mm * MM * rng.uniform(0.4, 1.6, count)[:, None]

    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float64) * 0.5
    tf = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]

    verts, faces = [], []
    for k in range(count):
        rot = _rand_rot(rng)
        verts.append(p[k] + (tet * s[k]) @ rot.T)
        faces.extend([[a + 4 * k for a in f] for f in tf])
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.concatenate(verts).tolist(), [], faces)
    me.validate()
    obj = bpy.data.objects.new(name, me)
    obj["jg4d_dust_home"] = p.ravel().tolist()
    obj["jg4d_dust_seed"] = int(seed)
    return obj


def _rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------

def fresh_mat(name):
    """Get-or-recreate by exact name. Without this, every rebuild leaves
    jg4d_glass_mat.001, .002, ... behind and the next lookup finds the stale one."""
    old = bpy.data.materials.get(name)
    if old is not None:
        try:
            bpy.data.materials.remove(old)
        except Exception:
            old.name = name + "_old"
    return bpy.data.materials.new(name)


def _nt(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return nt


def _mix_color(nt, blend, x, y):
    """A color Mix node across API generations, returning its sockets.

    ShaderNodeMixRGB is legacy in 4.x and a removal candidate; ShaderNodeMix
    (3.4+) replaces it but overloads socket NAMES ("A" is three different
    sockets), so the color sockets must be taken by INDEX: 6/7 in, 2 out.
    Returns (node, fac_socket, a_socket, b_socket, out_socket)."""
    try:
        n = nt.nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        n.location = (x, y)
        return n, n.inputs["Factor"], n.inputs[6], n.inputs[7], n.outputs[2]
    except RuntimeError:
        n = nt.nodes.new("ShaderNodeMixRGB")
        n.blend_type = blend
        n.location = (x, y)
        return n, n.inputs["Fac"], n.inputs["Color1"], n.inputs["Color2"], n.outputs["Color"]


def _n(nt, t, x, y):
    node = nt.nodes.new(t)
    node.location = (x, y)
    return node


def make_paper_material(name, cfg, props):
    """Albedo + alpha from the scan; ink drives bump AND gloss AND the rebuilt
    halftone; paper tooth is procedural because 300 ppi cannot resolve fibre."""
    paths = cfg["paths"]
    mat = fresh_mat(name)
    set_blend(mat, "BLEND")
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = "CLIP"
        except TypeError:
            pass
    nt = _nt(mat)

    uv = _n(nt, "ShaderNodeUVMap", -1500, 0)
    uv.uv_map = "UVMap"

    tex_alb = _n(nt, "ShaderNodeTexImage", -1250, 250)
    tex_alb.image = load_image(paths["albedo"])
    tex_alb.extension = "EXTEND"

    tex_ink = _n(nt, "ShaderNodeTexImage", -1250, -60)
    tex_ink.image = load_image(paths["ink"], noncolor=True)
    tex_ink.extension = "EXTEND"

    tex_alp = _n(nt, "ShaderNodeTexImage", -1250, -370)
    tex_alp.image = load_image(paths["alpha"], noncolor=True)
    tex_alp.extension = "EXTEND"

    for t in (tex_alb, tex_ink, tex_alp):
        if t.image:
            nt.links.new(uv.outputs["UV"], t.inputs["Vector"])

    # --- rebuilt halftone -----------------------------------------------------
    # screen = sin(2*pi*n*u) * sin(2*pi*n*v), remapped to 0..1: a smooth lattice
    # peaking at each cell centre. Threshold it against ink density and the dot
    # AREA tracks density -- which is exactly what an AM halftone is. Unlike the
    # scan, Cycles resolves this with its pixel filter, so it anti-aliases.
    w_mm, h_mm = cfg["physical_mm"]
    ruling = props.halftone_ruling
    cells_u = (w_mm / 25.4) * ruling
    cells_v = (h_mm / 25.4) * ruling

    m_ht = _n(nt, "ShaderNodeMapping", -1000, -700)
    m_ht.inputs["Rotation"].default_value = (0, 0, math.radians(props.halftone_angle))
    m_ht.inputs["Scale"].default_value = (cells_u, cells_v, 1.0)
    nt.links.new(uv.outputs["UV"], m_ht.inputs["Vector"])

    sep = _n(nt, "ShaderNodeSeparateXYZ", -820, -700)
    nt.links.new(m_ht.outputs["Vector"], sep.inputs["Vector"])

    sx = _n(nt, "ShaderNodeMath", -650, -620); sx.operation = "MULTIPLY"
    sx.inputs[1].default_value = 2.0 * math.pi
    sy = _n(nt, "ShaderNodeMath", -650, -780); sy.operation = "MULTIPLY"
    sy.inputs[1].default_value = 2.0 * math.pi
    nt.links.new(sep.outputs["X"], sx.inputs[0])
    nt.links.new(sep.outputs["Y"], sy.inputs[0])

    nx = _n(nt, "ShaderNodeMath", -480, -620); nx.operation = "SINE"
    ny = _n(nt, "ShaderNodeMath", -480, -780); ny.operation = "SINE"
    nt.links.new(sx.outputs[0], nx.inputs[0])
    nt.links.new(sy.outputs[0], ny.inputs[0])

    scr = _n(nt, "ShaderNodeMath", -310, -700); scr.operation = "MULTIPLY"
    nt.links.new(nx.outputs[0], scr.inputs[0])
    nt.links.new(ny.outputs[0], scr.inputs[1])

    scr01 = _n(nt, "ShaderNodeMapRange", -150, -700)
    scr01.inputs["From Min"].default_value = -1.0
    scr01.inputs["From Max"].default_value = 1.0
    nt.links.new(scr.outputs[0], scr01.inputs["Value"])

    inv = _n(nt, "ShaderNodeMath", -150, -900); inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    if tex_ink.image:
        nt.links.new(tex_ink.outputs["Color"], inv.inputs[1])
    else:
        inv.inputs[1].default_value = 0.0

    dot = _n(nt, "ShaderNodeMath", 30, -800); dot.operation = "GREATER_THAN"
    nt.links.new(scr01.outputs["Result"], dot.inputs[0])
    nt.links.new(inv.outputs[0], dot.inputs[1])

    dot_amt = _n(nt, "ShaderNodeMath", 200, -800); dot_amt.operation = "MULTIPLY"
    dot_amt.inputs[1].default_value = props.halftone_amount
    nt.links.new(dot.outputs[0], dot_amt.inputs[0])

    # --- paper tooth ----------------------------------------------------------
    tex_co = _n(nt, "ShaderNodeTexCoord", -1500, -1150)
    m_tooth = _n(nt, "ShaderNodeMapping", -1300, -1150)
    tooth_scale = (w_mm / 25.4) * 220.0     # ~220 "fibres" per inch
    m_tooth.inputs["Scale"].default_value = (tooth_scale, tooth_scale, tooth_scale)
    nt.links.new(tex_co.outputs["Object"], m_tooth.inputs["Vector"])

    tooth = _n(nt, "ShaderNodeTexNoise", -1100, -1150)
    set_input(tooth, "Scale", 8.0)
    set_input(tooth, "Detail", 6.0)
    set_input(tooth, "Roughness", 0.62)
    nt.links.new(m_tooth.outputs["Vector"], tooth.inputs["Vector"])

    # --- bump: ink relief + dots + tooth --------------------------------------
    relief = _n(nt, "ShaderNodeMath", 200, -1000); relief.operation = "ADD"
    if tex_ink.image:
        nt.links.new(tex_ink.outputs["Color"], relief.inputs[0])
    else:
        relief.inputs[0].default_value = 0.0
    nt.links.new(dot_amt.outputs[0], relief.inputs[1])

    bump_mix, bm_fac, bm_a, bm_b, bm_out = _mix_color(nt, "ADD", 380, -1050)
    bm_fac.default_value = 0.5
    nt.links.new(relief.outputs[0], bm_a)
    nt.links.new(tooth.outputs["Fac"], bm_b)

    bump = _n(nt, "ShaderNodeBump", 560, -1050)
    set_input(bump, "Strength", props.paper_bump)
    set_input(bump, "Distance", 0.06 * MM)
    nt.links.new(bm_out, bump.inputs["Height"])

    # --- roughness: ink is smoother than the sheet it sits on ------------------
    rough = _n(nt, "ShaderNodeMapRange", 380, -400)
    rough.inputs["From Min"].default_value = 0.0
    rough.inputs["From Max"].default_value = 1.0
    rough.inputs["To Min"].default_value = props.paper_roughness
    rough.inputs["To Max"].default_value = max(0.05, props.paper_roughness - 0.32)
    ink_for_rough = _n(nt, "ShaderNodeMath", 200, -400)
    ink_for_rough.operation = "MAXIMUM"
    if tex_ink.image:
        nt.links.new(tex_ink.outputs["Color"], ink_for_rough.inputs[0])
    else:
        ink_for_rough.inputs[0].default_value = 0.0
    nt.links.new(dot_amt.outputs[0], ink_for_rough.inputs[1])
    nt.links.new(ink_for_rough.outputs[0], rough.inputs["Value"])

    # tooth also breaks up roughness a little, or the sheet looks laminated
    rough_var, rv_fac, rv_a, rv_b, rv_out = _mix_color(nt, "MIX", 560, -400)
    rv_fac.default_value = 0.12
    nt.links.new(rough.outputs["Result"], rv_a)
    nt.links.new(tooth.outputs["Fac"], rv_b)

    # --- albedo, optionally darkened in the dots -------------------------------
    dot_dark, dd_fac, dd_a, dd_b, dd_out = _mix_color(nt, "MULTIPLY", 380, 250)
    dd_fac.default_value = props.halftone_albedo
    dd_b.default_value = (0.35, 0.32, 0.30, 1.0)
    if tex_alb.image:
        nt.links.new(tex_alb.outputs["Color"], dd_a)
    else:
        dd_a.default_value = tuple(cfg["paper_rgb"]) + (1.0,)
    fac_link = _n(nt, "ShaderNodeMath", 200, 100); fac_link.operation = "MULTIPLY"
    fac_link.inputs[1].default_value = props.halftone_albedo
    nt.links.new(dot.outputs[0], fac_link.inputs[0])
    nt.links.new(fac_link.outputs[0], dd_fac)

    # --- the surface itself ----------------------------------------------------
    # No emission. The museum feel is that it is LIT, not that it glows -- an
    # emissive item would also sit outside the volume's lighting and read as a
    # sticker pasted on the black rather than an object inside the case.
    bsdf = _n(nt, "ShaderNodeBsdfPrincipled", 760, 100)
    nt.links.new(dd_out, bsdf.inputs["Base Color"])
    nt.links.new(rv_out, bsdf.inputs["Roughness"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    set_input(bsdf, ("Specular IOR Level", "Specular"), 0.35)
    set_input(bsdf, ("Sheen Weight", "Sheen"), 0.12)   # paper has a soft sheen
    set_input(bsdf, ("Sheen Roughness",), 0.5)

    out = _n(nt, "ShaderNodeOutputMaterial", 1180, 100)

    if tex_alp.image:
        transp = _n(nt, "ShaderNodeBsdfTransparent", 760, -180)
        mix = _n(nt, "ShaderNodeMixShader", 980, 100)
        nt.links.new(tex_alp.outputs["Color"], mix.inputs["Fac"])
        nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
        nt.links.new(bsdf.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_glass_material(name, cfg, props):
    """Procedural scuffs so Randomize glass is instant and file-free. A painted
    map in the scuff slot overrides them when you want authored wear.

    Two tricks make glass read against black at all: the roughness lift in the
    wipes (so a grazing light draws them), and the world's reflection-only HDRI
    (so the pane carries a room the camera never sees)."""
    mat = fresh_mat(name)
    set_blend(mat, "BLEND")
    nt = _nt(mat)
    rng = np.random.default_rng(int(props.seed_scuff))

    tc = _n(nt, "ShaderNodeTexCoord", -1900, 0)
    # Normalise object space to "fractions of the pane" so the wear pattern does
    # not change scale when you resize the case. Everything downstream is in
    # pane units, which is also how you would describe a real scuff.
    co = _n(nt, "ShaderNodeMapping", -1750, 0)
    pw = max(props.case_width_mm * MM, 1e-6)
    ph = max(props.case_height_mm * MM, 1e-6)
    co.inputs["Scale"].default_value = (1.0 / pw, 1.0 / pw, 1.0 / ph)
    nt.links.new(tc.outputs["Object"], co.inputs["Vector"])

    # wipe arcs: anisotropic noise = streaks. Bending the coordinate with a
    # radial gradient turns straight streaks into arcs, which is how a cloth in
    # a hand actually moves.
    grad = _n(nt, "ShaderNodeTexGradient", -1400, -320)
    grad.gradient_type = "SPHERICAL"
    m_g = _n(nt, "ShaderNodeMapping", -1560, -320)
    m_g.inputs["Location"].default_value = (
        float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.6, 0.6)), 0.0)
    nt.links.new(co.outputs["Vector"], m_g.inputs["Vector"])
    nt.links.new(m_g.outputs["Vector"], grad.inputs["Vector"])

    bend, be_fac, be_a, be_b, be_out = _mix_color(nt, "ADD", -1200, -180)
    be_fac.default_value = float(rng.uniform(0.35, 0.8))
    nt.links.new(co.outputs["Vector"], be_a)
    nt.links.new(grad.outputs["Color"], be_b)

    m_w = _n(nt, "ShaderNodeMapping", -1000, -180)
    m_w.inputs["Rotation"].default_value = (0, 0, float(rng.uniform(0, math.pi)))
    m_w.inputs["Scale"].default_value = (1.0, float(rng.uniform(18, 42)), 1.0)
    nt.links.new(be_out, m_w.inputs["Vector"])

    wipe = _n(nt, "ShaderNodeTexNoise", -820, -180)
    set_input(wipe, "Scale", float(rng.uniform(3.0, 6.0)))
    set_input(wipe, "Detail", 3.0)
    set_input(wipe, "Roughness", 0.5)
    nt.links.new(m_w.outputs["Vector"], wipe.inputs["Vector"])

    wipe_r = _n(nt, "ShaderNodeValToRGB", -640, -180)
    wipe_r.color_ramp.elements[0].position = 0.52
    wipe_r.color_ramp.elements[1].position = 0.62
    nt.links.new(wipe.outputs["Fac"], wipe_r.inputs["Fac"])

    # hard scratches: Voronoi distance-to-edge is a line network; a tight ramp
    # keeps only the lines, anisotropic scaling gives them a direction.
    m_s = _n(nt, "ShaderNodeMapping", -1000, 180)
    m_s.inputs["Rotation"].default_value = (0, 0, float(rng.uniform(0, math.pi)))
    m_s.inputs["Scale"].default_value = (float(rng.uniform(0.6, 1.4)),
                                         float(rng.uniform(6, 16)), 1.0)
    nt.links.new(co.outputs["Vector"], m_s.inputs["Vector"])

    scr = _n(nt, "ShaderNodeTexVoronoi", -820, 180)
    scr.feature = "DISTANCE_TO_EDGE"
    set_input(scr, "Scale", float(rng.uniform(6.0, 14.0)))
    nt.links.new(m_s.outputs["Vector"], scr.inputs["Vector"])

    scr_r = _n(nt, "ShaderNodeValToRGB", -640, 180)
    scr_r.color_ramp.elements[0].position = 0.0
    scr_r.color_ramp.elements[1].position = float(rng.uniform(0.012, 0.03))
    scr_r.color_ramp.elements[0].color = (1, 1, 1, 1)
    scr_r.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(scr.outputs["Distance"], scr_r.inputs["Fac"])

    # smudges: low frequency, soft, barely there
    smudge = _n(nt, "ShaderNodeTexNoise", -820, 480)
    set_input(smudge, "Scale", float(rng.uniform(1.4, 3.0)))
    set_input(smudge, "Detail", 2.0)
    nt.links.new(co.outputs["Vector"], smudge.inputs["Vector"])
    smudge_r = _n(nt, "ShaderNodeValToRGB", -640, 480)
    smudge_r.color_ramp.elements[0].position = 0.55
    smudge_r.color_ramp.elements[1].position = 0.85
    nt.links.new(smudge.outputs["Fac"], smudge_r.inputs["Fac"])

    mix1, m1_fac, m1_a, m1_b, m1_out = _mix_color(nt, "ADD", -420, 0)
    m1_fac.default_value = 1.0
    nt.links.new(wipe_r.outputs["Color"], m1_a)
    nt.links.new(scr_r.outputs["Color"], m1_b)

    mix2, m2_fac, m2_a, m2_b, m2_out = _mix_color(nt, "ADD", -240, 0)
    m2_fac.default_value = 0.35
    nt.links.new(m1_out, m2_a)
    nt.links.new(smudge_r.outputs["Color"], m2_b)

    scuff_src = m2_out

    # painted override
    if props.use_scuff_map and cfg["paths"].get("scuff"):
        img = load_image(cfg["paths"]["scuff"], noncolor=True)
        if img:
            uv = _n(nt, "ShaderNodeUVMap", -1600, 700)
            uv.uv_map = "UVMap"
            tex = _n(nt, "ShaderNodeTexImage", -1400, 700)
            tex.image = img
            tex.extension = "EXTEND"
            nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
            over, ov_fac, ov_a, ov_b, ov_out = _mix_color(nt, "MIX", -240, 700)
            ov_fac.default_value = 1.0
            nt.links.new(m2_out, ov_a)
            nt.links.new(tex.outputs["Color"], ov_b)
            scuff_src = ov_out

    amt = _n(nt, "ShaderNodeMath", -60, 0); amt.operation = "MULTIPLY"
    amt.inputs[1].default_value = props.scuff_amount
    nt.links.new(scuff_src, amt.inputs[0])

    rough = _n(nt, "ShaderNodeMapRange", 120, -200)
    rough.inputs["To Min"].default_value = props.glass_roughness
    rough.inputs["To Max"].default_value = props.scuff_roughness
    nt.links.new(amt.outputs[0], rough.inputs["Value"])

    bump = _n(nt, "ShaderNodeBump", 120, -450)
    set_input(bump, "Strength", 0.22)
    set_input(bump, "Distance", 0.0004)
    nt.links.new(amt.outputs[0], bump.inputs["Height"])

    bsdf = _n(nt, "ShaderNodeBsdfPrincipled", 420, 0)
    # Principled has no separate transmission colour in either 3.x or 4.x:
    # with Transmission at 1, Base Color IS the tint of the transmitted light.
    # The faint green is what says "real glass" rather than "hole in space" --
    # soda-lime is iron-tinted and a case pane is thick enough to show it.
    set_input(bsdf, "Base Color", (0.92, 0.98, 0.94, 1.0))
    set_input(bsdf, ("Transmission Weight", "Transmission"), 1.0)
    set_input(bsdf, "IOR", 1.52)
    set_input(bsdf, ("Metallic",), 0.0)
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    out = _n(nt, "ShaderNodeOutputMaterial", 700, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_velvet_material(name, albedo=0.03):
    mat = fresh_mat(name)
    nt = _nt(mat)
    b = _n(nt, "ShaderNodeBsdfPrincipled", 0, 0)
    set_input(b, "Base Color", (albedo, albedo * 0.97, albedo * 1.02, 1.0))
    set_input(b, "Roughness", 0.95)
    set_input(b, ("Sheen Weight", "Sheen"), 0.55)
    set_input(b, ("Specular IOR Level", "Specular"), 0.08)
    out = _n(nt, "ShaderNodeOutputMaterial", 260, 0)
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_dust_material(name):
    mat = fresh_mat(name)
    nt = _nt(mat)
    b = _n(nt, "ShaderNodeBsdfPrincipled", 0, 0)
    set_input(b, "Base Color", (0.85, 0.84, 0.80, 1.0))
    set_input(b, "Roughness", 0.9)
    set_input(b, ("Specular IOR Level", "Specular"), 0.2)
    out = _n(nt, "ShaderNodeOutputMaterial", 260, 0)
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_volume_material(name, density, anisotropy, color=(0.85, 0.87, 0.95)):
    mat = fresh_mat(name)
    nt = _nt(mat)
    v = _n(nt, "ShaderNodeVolumeScatter", 0, 0)
    set_input(v, "Color", tuple(color) + (1.0,))
    set_input(v, "Density", density)
    set_input(v, "Anisotropy", anisotropy)
    out = _n(nt, "ShaderNodeOutputMaterial", 260, 0)
    nt.links.new(v.outputs["Volume"], out.inputs["Volume"])
    return mat


# ----------------------------------------------------------------------------
# World: black to the camera, a room to the reflections
#
# Is Camera Ray is 1 only for the primary ray. Mix black on that and the HDRI
# otherwise, and the camera sees void while the glass still reflects a room and
# the volume still gets a whisper of ambient. This is the node that makes the
# pane exist.
# ----------------------------------------------------------------------------

def setup_world(scene, props):
    w = bpy.data.worlds.get("jg4d_vitrine_world") or bpy.data.worlds.new("jg4d_vitrine_world")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()

    out = _n(nt, "ShaderNodeOutputWorld", 500, 0)
    lp = _n(nt, "ShaderNodeLightPath", -400, 300)
    black = _n(nt, "ShaderNodeBackground", 100, -150)
    set_input(black, "Color", (0, 0, 0, 1))
    set_input(black, "Strength", 0.0)

    room = _n(nt, "ShaderNodeBackground", 100, 150)
    set_input(room, "Strength", props.room_strength)

    if props.room_hdri and os.path.isfile(bpy.path.abspath(props.room_hdri)):
        env = _n(nt, "ShaderNodeTexEnvironment", -180, 150)
        env.image = load_image(bpy.path.abspath(props.room_hdri))
        m = _n(nt, "ShaderNodeMapping", -380, 150)
        m.inputs["Rotation"].default_value = (0, 0, props.room_rotation)
        tc = _n(nt, "ShaderNodeTexCoord", -560, 150)
        nt.links.new(tc.outputs["Generated"], m.inputs["Vector"])
        nt.links.new(m.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], room.inputs["Color"])
    else:
        # No HDRI: a cheap vertical gradient still gives the pane something to
        # reflect, which is all we need -- an unbroken specular is what makes CG
        # glass look like nothing at all.
        grad = _n(nt, "ShaderNodeTexGradient", -180, 150)
        ramp = _n(nt, "ShaderNodeValToRGB", -20, 150)
        ramp.color_ramp.elements[0].color = (0.02, 0.02, 0.03, 1)
        ramp.color_ramp.elements[1].color = (0.55, 0.58, 0.68, 1)
        m = _n(nt, "ShaderNodeMapping", -380, 150)
        m.inputs["Rotation"].default_value = (math.radians(90), 0, props.room_rotation)
        tc = _n(nt, "ShaderNodeTexCoord", -560, 150)
        nt.links.new(tc.outputs["Generated"], m.inputs["Vector"])
        nt.links.new(m.outputs["Vector"], grad.inputs["Vector"])
        nt.links.new(grad.outputs["Color"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], room.inputs["Color"])

    mix = _n(nt, "ShaderNodeMixShader", 320, 0)
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(room.outputs["Background"], mix.inputs[1])
    nt.links.new(black.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return w


# ----------------------------------------------------------------------------
# Lights
# ----------------------------------------------------------------------------

def kelvin_to_rgb(k):
    """Tanner Helland's approximation. Good enough for a light tint, and it
    keeps the panel in Kelvin, which is how you would actually order a fixture."""
    t = k / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(max(t, 1e-3)) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(max(t - 10, 1e-3)) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255.0
    return tuple(min(1.0, max(0.0, v / 255.0)) for v in (r, g, b))


def build_lights(props, cfg, case, seed):
    """Three fixtures, and each is doing one job:
       key   -- a case puck: small, steep, hard-ish. Makes the cone in the volume.
       fill  -- cool, dim, wide: keeps the shadow side from going to pure black,
                which in anaglyph is where the eyes lose the object entirely.
       sheen -- long, thin, grazing: this one exists ONLY to sweep the glass and
                the paper tooth. It is the light that draws the scuffs.
    """
    rng = np.random.default_rng(int(seed))
    v = float(props.var_light)
    cw, ch, cd = case["w"], case["h"], case["d"]
    out = []

    def jit(base, frac):
        return base * (1.0 + float(rng.uniform(-frac, frac)) * v)

    kd = bpy.data.lights.new("jg4d_key", "AREA")
    kd.shape = "DISK"
    kd.size = jit(0.035, 0.35)
    kd.energy = jit(props.key_energy, 0.25)
    kd.color = kelvin_to_rgb(jit(props.key_kelvin, 0.06))
    key = bpy.data.objects.new("jg4d_key", kd)
    key.location = (jit(0.10, 0.6) * rng.choice([-1, 1]),
                    -cd * 0.18 + jit(0.02, 1.0),
                    ch * 0.5 - 0.02)
    key.rotation_euler = (math.radians(jit(18, 0.5)), 0, float(rng.uniform(0, math.pi)))
    out.append(("key", key))

    fd = bpy.data.lights.new("jg4d_fill", "AREA")
    fd.shape = "RECTANGLE"
    fd.size, fd.size_y = cw * 0.8, cd * 0.8
    fd.energy = jit(props.fill_energy, 0.3)
    fd.color = kelvin_to_rgb(jit(props.fill_kelvin, 0.05))
    fill = bpy.data.objects.new("jg4d_fill", fd)
    fill.location = (jit(-0.12, 0.4), cd * 0.3, ch * 0.2)
    fill.rotation_euler = (math.radians(jit(65, 0.2)), 0, math.radians(jit(-25, 0.5)))
    out.append(("fill", fill))

    sd = bpy.data.lights.new("jg4d_sheen", "AREA")
    sd.shape = "RECTANGLE"
    sd.size, sd.size_y = cw * 1.6, 0.01
    sd.energy = jit(props.sheen_energy, 0.3)
    sd.color = kelvin_to_rgb(jit(props.sheen_kelvin, 0.05))
    sheen = bpy.data.objects.new("jg4d_sheen", sd)
    side = rng.choice([-1.0, 1.0])
    sheen.location = (jit(cw * 0.9, 0.2) * side, -cd * 0.55, jit(ch * 0.15, 0.6))
    sheen.rotation_euler = (math.radians(jit(88, 0.05)), 0,
                            math.radians(jit(90, 0.06)) * side)
    out.append(("sheen", sheen))
    return out


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def clear_rig(context, keep=("camera",)):
    """Remove the rig and its orphaned data. Keeps the camera by default, so a
    rebuild does not throw away a shot you already framed."""
    for obj in list(context.scene.collection.all_objects):
        kind = obj.get(TAG)
        if kind is None or kind in keep:
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is None or data.users:
            continue
        try:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Light):
                bpy.data.lights.remove(data)
        except Exception:
            pass


def build_vitrine(context, props):
    scene = context.scene
    cfg = load_sidecar(bpy.path.abspath(props.albedo_path))

    w_mm, h_mm = cfg["physical_mm"]
    if props.override_size:
        w_mm = props.width_mm
        h_mm = props.width_mm * (cfg["physical_mm"][1] / max(cfg["physical_mm"][0], 1e-6))
        cfg["physical_mm"] = [w_mm, h_mm]

    clear_rig(context)

    root = bpy.data.objects.new("jg4d_vitrine", None)
    root.empty_display_size = 0.05
    root[TAG] = "root"
    scene.collection.objects.link(root)

    case = {"w": props.case_width_mm * MM,
            "h": props.case_height_mm * MM,
            "d": props.case_depth_mm * MM}

    def add(obj, kind, parent=root):
        obj[TAG] = kind
        scene.collection.objects.link(obj)
        obj.parent = parent
        return obj

    # Camera first: the safe ruling depends on how big the item actually lands
    # in frame, and the paper material needs the ruling before it is built.
    cam = ensure_camera(context, props, case)
    if props.auto_safe_ruling:
        Z = props.cam_distance_mm * MM + case["d"] * 0.5 + props.item_depth_mm * MM
        frame_h_mm = 2.0 * Z * math.tan(vfov_of(cam, scene) / 2.0) / MM
        px = scene.render.resolution_y * (h_mm / max(frame_h_mm, 1e-6))
        props.halftone_ruling = round(safe_ruling(h_mm, px, 3.0), 1)
        scr = cfg.get("screen")
        if isinstance(scr, dict) and scr.get("angle_deg") is not None:
            props.halftone_angle = float(scr["angle_deg"])

    # --- the item -------------------------------------------------------------
    paper = build_paper("jg4d_item", w_mm, h_mm, props.seed_curl, props.curl_mm,
                        res=props.paper_res)
    paper.data.materials.append(make_paper_material("jg4d_item_mat", cfg, props))
    paper.rotation_euler = (math.radians(90 - props.item_tilt), 0, 0)
    paper.location = (0, props.item_depth_mm * MM, 0)
    add(paper, "item")

    # --- pedestal: a dark shelf, not a void ------------------------------------
    # 2-4% albedo velvet. Its lit top edge is what tells the eye the item is
    # standing in a space rather than floating in nothing -- and in stereo a
    # single receding surface does more for depth than any amount of parallax.
    ped = build_box("jg4d_pedestal", case["w"] * 0.98, 0.02, case["d"] * 0.98,
                    center=(0, 0, -case["h"] * 0.5 - 0.01))
    ped.data.materials.append(make_velvet_material("jg4d_velvet", props.pedestal_albedo))
    add(ped, "pedestal")

    back = build_plane("jg4d_back", case["w"] * 0.98, case["h"] * 0.98,
                       center=(0, case["d"] * 0.5, 0), axis="XZ")
    back.data.materials.append(ped.data.materials[0])
    add(back, "back")

    # --- the glass ------------------------------------------------------------
    glass_y = -case["d"] * 0.5
    glass = build_box("jg4d_glass", case["w"], case["h"], props.glass_thickness_mm * MM,
                      center=(0, glass_y, 0))
    box_uvs(glass.data, case["w"], case["h"])
    glass.data.materials.append(make_glass_material("jg4d_glass_mat", cfg, props))
    add(glass, "glass")

    # --- volume ---------------------------------------------------------------
    vol = build_box("jg4d_volume", case["w"] * 0.99, case["h"] * 0.99, case["d"] * 0.96,
                    center=(0, 0.01 * case["d"], 0))
    vol.data.materials.append(make_volume_material(
        "jg4d_volume_mat", props.volume_density, props.volume_anisotropy))
    vol.display_type = "WIRE"
    add(vol, "volume")

    # --- dust -----------------------------------------------------------------
    # Kept strictly behind the glass. A mote in front of the pane would sit in
    # negative parallax against a screen edge and that is a window violation --
    # the one artefact anaglyph punishes hardest.
    dust = build_dust("jg4d_dust", props.dust_count,
                      case["w"] * 0.9, case["h"] * 0.9, case["d"] * 0.8,
                      props.seed_dust, center=(0, 0.02 * case["d"], 0),
                      size_mm=props.dust_size_mm)
    dust.data.materials.append(make_dust_material("jg4d_dust_mat"))
    add(dust, "dust")

    # --- lights ---------------------------------------------------------------
    for kind, obj in build_lights(props, cfg, case, props.seed_light):
        add(obj, "light_" + kind)

    # --- world ----------------------------------------------------------------
    setup_world(scene, props)

    root[TAG + "_cfg"] = json.dumps({
        "physical_mm": [w_mm, h_mm],
        "case": {k: v / MM for k, v in case.items()},
        "glass_y_mm": glass_y / MM,
        "albedo": props.albedo_path,
    })
    return root, cfg, case


def ensure_camera(context, props, case):
    scene = context.scene
    cam = None
    for o in scene.collection.all_objects:
        if o.type == "CAMERA" and o.get(TAG) == "camera":
            cam = o
            break
    fresh = cam is None
    if fresh:
        cd = bpy.data.cameras.new("jg4d_vitrine_cam")
        cam = bpy.data.objects.new("jg4d_vitrine_cam", cd)
        cam[TAG] = "camera"
        scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens = props.lens_mm
    if fresh or props.reset_camera:
        cam.location = (0, -case["d"] * 0.5 - props.cam_distance_mm * MM, 0)
        cam.rotation_euler = (math.radians(90), 0, 0)  # look down +Y at the case
    return cam


# ----------------------------------------------------------------------------
# Dust drift
#
# frame_change_post, exactly like the LDI3 player's re-displace. Motes wander on
# a slow seeded curl field. Deterministic in the frame number, so a re-render of
# frame 812 next month is identical -- no cache, no bake, no sim.
# ----------------------------------------------------------------------------

def _drift_offsets(home, t, seed, amp, speed):
    """Seeded curl-ish field sampled at each mote's home position.

    Spatial frequency is ~100 rad/m on purpose: motes within a few centimetres
    drift together (air moves in parcels, not as independent particles) while
    motes across the case decorrelate. Too low and the whole cloud slides as one
    rigid lump; too high and it boils, which in stereo is per-eye noise.

    `amp` is a hard bound -- the vertical term shares the budget rather than
    adding to it, so "Drift amp 4 mm" cannot put a mote 5.4 mm from home. That
    matters: the dust staying behind the glass is what makes the no-window-
    violation guarantee true rather than merely likely.
    """
    rng = np.random.default_rng(int(seed) ^ 0x5EED)
    ph = rng.uniform(0, 2 * math.pi, (3, 3))
    fr = rng.uniform(0.6, 1.8, (3, 3))
    x, y, z = home[:, 0], home[:, 1], home[:, 2]
    o = np.zeros_like(home)
    for k, (a, b) in enumerate(((y, z), (z, x), (x, y))):
        o[:, k] = (np.sin(a * 110.0 * fr[k, 0] + t * speed * fr[k, 1] + ph[k, 0]) *
                   np.cos(b * 85.0 * fr[k, 1] + t * speed * fr[k, 2] + ph[k, 2]))
    # a slow common rise, inside the budget rather than on top of it
    o[:, 2] = 0.72 * o[:, 2] + 0.28 * np.sin(t * speed * 0.31 + ph[2, 2])
    return o * amp


def refresh_dust(obj, scene):
    home = np.array(obj["jg4d_dust_home"], np.float64).reshape(-1, 3)
    seed = int(obj["jg4d_dust_seed"])
    props = scene.jg4d_vitrine
    t = scene.frame_current / max(scene.render.fps, 1)
    off = _drift_offsets(home, t, seed, props.dust_drift_mm * MM, props.dust_speed)

    me = obj.data
    n = len(me.vertices)
    per = n // max(len(home), 1)
    if per <= 0:
        return
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    base = obj.get("jg4d_dust_base")
    if base is None:
        obj["jg4d_dust_base"] = co.ravel().tolist()
        base = obj["jg4d_dust_base"]
    base = np.array(base, np.float64).reshape(-1, 3)
    co = base + np.repeat(off, per, axis=0)
    me.vertices.foreach_set("co", co.ravel().astype(np.float32))
    me.update()


def jg4d_vitrine_frame_handler(scene, _depsgraph=None):
    props = getattr(scene, "jg4d_vitrine", None)
    if props is None or not props.dust_animate:
        return
    for obj in scene.collection.all_objects:
        if obj.get(TAG) == "dust" and "jg4d_dust_home" in obj:
            refresh_dust(obj, scene)


# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------

class JG4D_VitrineProps(bpy.types.PropertyGroup):
    albedo_path: bpy.props.StringProperty(
        name="Item", subtype="FILE_PATH",
        description="Any *_albedo.png from vitrine_prep.py; siblings and sidecar are found automatically")

    override_size: bpy.props.BoolProperty(
        name="Override size", default=False,
        description="Ignore the sidecar's physical size (from scan ppi) and set the width by hand")
    width_mm: bpy.props.FloatProperty(name="Width", default=180.0, min=5.0, max=2000.0, unit="NONE")

    # case
    case_width_mm: bpy.props.FloatProperty(name="Case W", default=340.0, min=20.0)
    case_height_mm: bpy.props.FloatProperty(name="Case H", default=420.0, min=20.0)
    case_depth_mm: bpy.props.FloatProperty(name="Case D", default=260.0, min=20.0)
    glass_thickness_mm: bpy.props.FloatProperty(name="Glass", default=5.0, min=0.5, max=30.0)
    item_depth_mm: bpy.props.FloatProperty(
        name="Item depth", default=40.0,
        description="How far behind the case centre the item sits. Bigger = more parallax")
    item_tilt: bpy.props.FloatProperty(name="Item tilt", default=8.0, min=-80, max=80)
    pedestal_albedo: bpy.props.FloatProperty(name="Velvet", default=0.03, min=0.0, max=0.2)

    # paper
    paper_res: bpy.props.IntProperty(name="Paper res", default=96, min=8, max=512)
    curl_mm: bpy.props.FloatProperty(name="Curl", default=3.0, min=0.0, max=40.0)
    paper_roughness: bpy.props.FloatProperty(name="Roughness", default=0.62, min=0.0, max=1.0)
    paper_bump: bpy.props.FloatProperty(name="Bump", default=0.35, min=0.0, max=1.0)

    # halftone
    halftone_ruling: bpy.props.FloatProperty(
        name="Ruling (lpi)", default=60.0, min=5.0, max=400.0,
        description="Use Safe ruling. Rebuilding the scan's true ruling re-creates the aliasing you removed")
    halftone_angle: bpy.props.FloatProperty(name="Angle", default=45.0, min=0.0, max=180.0)
    halftone_amount: bpy.props.FloatProperty(
        name="Dot relief", default=0.55, min=0.0, max=1.0,
        description="How much the dots bump and gloss. This is the one that makes ink read as ink")
    halftone_albedo: bpy.props.FloatProperty(
        name="Dot in color", default=0.12, min=0.0, max=1.0,
        description="How much the dots darken the albedo. Keep low: the scan already has the tone")

    # glass
    glass_roughness: bpy.props.FloatProperty(name="Base rough", default=0.02, min=0.0, max=0.3)
    scuff_roughness: bpy.props.FloatProperty(name="Scuff rough", default=0.20, min=0.0, max=1.0)
    scuff_amount: bpy.props.FloatProperty(name="Scuff amount", default=0.7, min=0.0, max=1.0)
    use_scuff_map: bpy.props.BoolProperty(
        name="Use painted scuff map", default=False,
        description="Override the procedural scuffs with the *_scuff.png next to the item")

    # volume + dust
    volume_density: bpy.props.FloatProperty(name="Haze", default=0.012, min=0.0, max=0.5)
    volume_anisotropy: bpy.props.FloatProperty(name="Anisotropy", default=0.4, min=-0.9, max=0.9)
    dust_count: bpy.props.IntProperty(name="Motes", default=400, min=0, max=6000)
    dust_size_mm: bpy.props.FloatProperty(name="Mote size", default=0.35, min=0.02, max=4.0)
    dust_animate: bpy.props.BoolProperty(name="Drift", default=True)
    dust_drift_mm: bpy.props.FloatProperty(name="Drift amp", default=4.0, min=0.0, max=60.0)
    dust_speed: bpy.props.FloatProperty(name="Drift speed", default=0.25, min=0.0, max=5.0)

    # lights
    key_energy: bpy.props.FloatProperty(name="Key W", default=12.0, min=0.0)
    key_kelvin: bpy.props.FloatProperty(name="Key K", default=3200.0, min=1000.0, max=12000.0)
    fill_energy: bpy.props.FloatProperty(name="Fill W", default=1.2, min=0.0)
    fill_kelvin: bpy.props.FloatProperty(name="Fill K", default=6500.0, min=1000.0, max=12000.0)
    sheen_energy: bpy.props.FloatProperty(name="Sheen W", default=6.0, min=0.0)
    sheen_kelvin: bpy.props.FloatProperty(name="Sheen K", default=4200.0, min=1000.0, max=12000.0)
    room_hdri: bpy.props.StringProperty(
        name="Room HDRI", subtype="FILE_PATH",
        description="Seen only in reflections. Empty = a built-in gradient, which is enough")
    room_strength: bpy.props.FloatProperty(name="Room", default=0.35, min=0.0, max=10.0)
    room_rotation: bpy.props.FloatProperty(name="Room rot", default=0.0, subtype="ANGLE")

    # camera / stereo
    auto_safe_ruling: bpy.props.BoolProperty(
        name="Auto safe ruling", default=True,
        description="On Build, set the halftone ruling to what the delivery frame can carry")
    reset_camera: bpy.props.BoolProperty(
        name="Reset camera on build", default=False,
        description="Off: a rebuild keeps the framing you set up")
    lens_mm: bpy.props.FloatProperty(name="Lens", default=70.0, min=8.0, max=400.0)
    cam_distance_mm: bpy.props.FloatProperty(name="Cam dist", default=420.0, min=20.0)
    target_parallax: bpy.props.FloatProperty(
        name="Max parallax %", default=1.0, min=0.05, max=4.0,
        description="Max on-screen parallax as % of frame width. Anaglyph wants ~1%; "
                    "polarised theatrical tolerates 2%+")

    # seeds + variation
    seed_curl: bpy.props.IntProperty(name="Curl seed", default=1)
    seed_scuff: bpy.props.IntProperty(name="Glass seed", default=1)
    seed_dust: bpy.props.IntProperty(name="Dust seed", default=1)
    seed_light: bpy.props.IntProperty(name="Light seed", default=1)
    var_light: bpy.props.FloatProperty(
        name="Light variation", default=0.25, min=0.0, max=1.0,
        description="How far Randomize lighting is allowed to wander. Low on purpose")


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------

def _find(context, kind):
    for o in context.scene.collection.all_objects:
        if o.get(TAG) == kind:
            return o
    return None


def _root_cfg(context):
    r = _find(context, "root")
    if r is None or TAG + "_cfg" not in r:
        return None, None
    return r, json.loads(r[TAG + "_cfg"])


class JG4D_OT_vitrine_build(bpy.types.Operator):
    """Build (or rebuild) the whole vitrine rig around the chosen item"""
    bl_idname = "jg4d.vitrine_build"
    bl_label = "Build vitrine"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.jg4d_vitrine
        p = bpy.path.abspath(props.albedo_path)
        if not p or not os.path.isfile(p):
            self.report({"ERROR"}, "Pick an *_albedo.png from vitrine_prep.py first")
            return {"CANCELLED"}
        try:
            root, cfg, case = build_vitrine(context, props)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, "vitrine build failed: %s" % e)
            return {"CANCELLED"}
        scene = context.scene
        scene.render.engine = "CYCLES"
        try:
            scene.cycles.volume_bounces = max(scene.cycles.volume_bounces, 2)
            scene.cycles.transmission_bounces = max(scene.cycles.transmission_bounces, 8)
        except Exception:
            pass
        self.report({"INFO"}, "Vitrine built: %.0f x %.0f mm item. Run Safe ruling, then Solve stereo."
                    % tuple(cfg["physical_mm"]))
        return {"FINISHED"}


class JG4D_OT_vitrine_safe_ruling(bpy.types.Operator):
    """Set the halftone ruling to the highest the delivery frame can carry"""
    bl_idname = "jg4d.vitrine_safe_ruling"
    bl_label = "Safe ruling"

    def execute(self, context):
        scene = context.scene
        props = scene.jg4d_vitrine
        root, cfg = _root_cfg(context)
        item = _find(context, "item")
        if not cfg or item is None:
            self.report({"ERROR"}, "Build the vitrine first")
            return {"CANCELLED"}
        h_mm = cfg["physical_mm"][1]

        # how tall the item actually lands in frame, at this lens and distance
        cam = _find(context, "camera") or scene.camera
        if cam is None:
            self.report({"ERROR"}, "No camera")
            return {"CANCELLED"}
        Z = abs(item.matrix_world.translation.y - cam.matrix_world.translation.y)
        vfov = 2 * math.atan(cam.data.sensor_height / (2 * cam.data.lens)) \
            if cam.data.sensor_fit == "VERTICAL" else \
            2 * math.atan((cam.data.sensor_width * scene.render.resolution_y /
                           max(scene.render.resolution_x, 1)) / (2 * cam.data.lens))
        frame_h_mm = 2.0 * Z * math.tan(vfov / 2.0) / MM
        px = scene.render.resolution_y * (h_mm / max(frame_h_mm, 1e-6))
        r = safe_ruling(h_mm, px, px_per_cell=3.0)

        true_lpi = cfg.get("screen", {}).get("ruling_lpi") if isinstance(cfg.get("screen"), dict) else None
        props.halftone_ruling = round(r, 1)
        msg = "Item lands %.0f px tall -> safe ruling %.0f lpi" % (px, r)
        if true_lpi:
            msg += " (scan's real screen was %.0f lpi; rebuilding that would give %.1f px/cell)" % (
                true_lpi, px / ((h_mm / 25.4) * true_lpi))
        self.report({"INFO"}, msg)
        print("jg4d vitrine: " + msg)
        return {"FINISHED"}


class JG4D_OT_vitrine_stereo(bpy.types.Operator):
    """Converge on the glass and solve the interaxial from the case geometry"""
    bl_idname = "jg4d.vitrine_solve_stereo"
    bl_label = "Solve stereo (converge on glass)"

    def execute(self, context):
        scene = context.scene
        props = scene.jg4d_vitrine
        root, cfg = _root_cfg(context)
        cam = _find(context, "camera") or scene.camera
        if not cfg or cam is None:
            self.report({"ERROR"}, "Build the vitrine first")
            return {"CANCELLED"}

        cam_y = cam.matrix_world.translation.y
        glass_y = cfg["glass_y_mm"] * MM
        back_y = cfg["case"]["d"] * MM * 0.5

        Zg = abs(glass_y - cam_y)                 # convergence: the pane
        Zb = abs(back_y - cam_y)                  # deepest thing in the case
        hfov = hfov_of(cam, scene)

        target = props.target_parallax / 100.0
        I_par = solve_interaxial(Zg, Zb, hfov, target)
        I_30 = Zg / 30.0                          # the old 1/30 rule of thumb
        I = min(I_par, I_30)

        scene.render.use_multiview = True
        scene.render.views_format = "STEREO_3D"
        scene.render.image_settings.views_format = "STEREO_3D"
        scene.render.image_settings.stereo_3d_format.display_mode = "ANAGLYPH"
        scene.render.image_settings.stereo_3d_format.anaglyph_type = "RED_CYAN"
        cam.data.stereo.convergence_mode = "OFFAXIS"
        cam.data.stereo.convergence_distance = Zg
        cam.data.stereo.interocular_distance = I

        p_back = parallax_frac(I, Zg, Zb, hfov) * 100.0
        item = _find(context, "item")
        p_item = 0.0
        if item:
            Zi = abs(item.matrix_world.translation.y - cam_y)
            p_item = parallax_frac(I, Zg, Zi, hfov) * 100.0

        msg = ("interaxial %.1f mm (parallax-limited %.1f, 1/30 rule %.1f) | "
               "converge %.0f mm on the glass | back of case %+.2f%%, item %+.2f%% of width"
               % (I / MM, I_par / MM, I_30 / MM, Zg / MM, p_back, p_item))
        self.report({"INFO"}, msg)
        print("jg4d vitrine stereo: " + msg)
        print("  hfov %.1f deg, lens %.0f mm, res %dx%d"
              % (math.degrees(hfov), cam.data.lens,
                 scene.render.resolution_x, scene.render.resolution_y))
        print("  Nothing sits in front of the glass, so parallax is positive "
              "everywhere: no window violation is possible by construction.")
        if I / MM > 40:
            print("  ! interaxial is large for a subject this size; the case may "
                  "read as a doll's house. Move the camera back or shorten the case.")
        return {"FINISHED"}


class _RandBase(bpy.types.Operator):
    bl_options = {"REGISTER", "UNDO"}

    def _need(self, context):
        if _find(context, "root") is None:
            self.report({"ERROR"}, "Build the vitrine first")
            return False
        return True


class JG4D_OT_rand_curl(_RandBase):
    """New paper curl. The sheet's bend, corner lift and ripple, reseeded"""
    bl_idname = "jg4d.vitrine_rand_curl"
    bl_label = "Curl"

    def execute(self, context):
        if not self._need(context):
            return {"CANCELLED"}
        props = context.scene.jg4d_vitrine
        props.seed_curl += 1
        root, cfg = _root_cfg(context)
        item = _find(context, "item")
        w, h = cfg["physical_mm"]
        if item and reshape_paper(item, w, h, props.seed_curl, props.curl_mm):
            self.report({"INFO"}, "curl seed %d" % props.seed_curl)
            return {"FINISHED"}
        self.report({"WARNING"}, "could not reshape; rebuild")
        return {"CANCELLED"}


class JG4D_OT_rand_scuff(_RandBase):
    """New glass wear. Different wipe arcs, scratches and smudges"""
    bl_idname = "jg4d.vitrine_rand_scuff"
    bl_label = "Glass"

    def execute(self, context):
        if not self._need(context):
            return {"CANCELLED"}
        props = context.scene.jg4d_vitrine
        props.seed_scuff += 1
        glass = _find(context, "glass")
        cfg = load_sidecar(bpy.path.abspath(props.albedo_path))
        if glass:
            old = glass.data.materials[0] if glass.data.materials else None
            mat = make_glass_material("jg4d_glass_mat_%d" % props.seed_scuff, cfg, props)
            glass.data.materials.clear()
            glass.data.materials.append(mat)
            if old and old.users == 0:
                bpy.data.materials.remove(old)
        self.report({"INFO"}, "glass seed %d" % props.seed_scuff)
        return {"FINISHED"}


class JG4D_OT_rand_dust(_RandBase):
    """New dust. Motes redistributed through the case volume"""
    bl_idname = "jg4d.vitrine_rand_dust"
    bl_label = "Dust"

    def execute(self, context):
        if not self._need(context):
            return {"CANCELLED"}
        props = context.scene.jg4d_vitrine
        props.seed_dust += 1
        root, cfg = _root_cfg(context)
        old = _find(context, "dust")
        case = {k: v * MM for k, v in cfg["case"].items()}
        new = build_dust("jg4d_dust", props.dust_count,
                         case["w"] * 0.9, case["h"] * 0.9, case["d"] * 0.8,
                         props.seed_dust, center=(0, 0.02 * case["d"], 0),
                         size_mm=props.dust_size_mm)
        mat = bpy.data.materials.get("jg4d_dust_mat") or make_dust_material("jg4d_dust_mat")
        new.data.materials.append(mat)
        new[TAG] = "dust"
        context.scene.collection.objects.link(new)
        new.parent = root
        if old:
            me = old.data
            bpy.data.objects.remove(old, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
        self.report({"INFO"}, "dust seed %d (%d motes)" % (props.seed_dust, props.dust_count))
        return {"FINISHED"}


class JG4D_OT_rand_light(_RandBase):
    """Nudge the lighting. Bounded by Light variation -- deliberately small"""
    bl_idname = "jg4d.vitrine_rand_light"
    bl_label = "Lighting"

    def execute(self, context):
        if not self._need(context):
            return {"CANCELLED"}
        props = context.scene.jg4d_vitrine
        props.seed_light += 1
        root, cfg = _root_cfg(context)
        case = {k: v * MM for k, v in cfg["case"].items()}
        for kind in ("key", "fill", "sheen"):
            o = _find(context, "light_" + kind)
            if o:
                d = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                if d.users == 0:
                    bpy.data.lights.remove(d)
        for kind, obj in build_lights(props, load_sidecar(bpy.path.abspath(props.albedo_path)),
                                      case, props.seed_light):
            obj[TAG] = "light_" + kind
            context.scene.collection.objects.link(obj)
            obj.parent = root
        self.report({"INFO"}, "light seed %d (variation %.0f%%)"
                    % (props.seed_light, props.var_light * 100))
        return {"FINISHED"}


class JG4D_OT_rand_all(_RandBase):
    """Reseed everything at once"""
    bl_idname = "jg4d.vitrine_rand_all"
    bl_label = "Randomize all"

    def execute(self, context):
        for op in (bpy.ops.jg4d.vitrine_rand_curl, bpy.ops.jg4d.vitrine_rand_scuff,
                   bpy.ops.jg4d.vitrine_rand_dust, bpy.ops.jg4d.vitrine_rand_light):
            try:
                op()
            except Exception as e:
                print("jg4d vitrine: randomize step failed: %s" % e)
        return {"FINISHED"}


# ----------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------

class JG4D_PT_vitrine(bpy.types.Panel):
    bl_label = "jg4d vitrine kit"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "jg4d"

    def draw(self, context):
        p = context.scene.jg4d_vitrine
        L = self.layout

        col = L.column()
        col.prop(p, "albedo_path", text="")
        col.operator("jg4d.vitrine_build", icon="MOD_BUILD")

        box = L.box()
        box.label(text="Randomize", icon="FILE_REFRESH")
        box.operator("jg4d.vitrine_rand_all", icon="SHADERFX")
        r = box.row(align=True)
        r.operator("jg4d.vitrine_rand_curl")
        r.operator("jg4d.vitrine_rand_scuff")
        r = box.row(align=True)
        r.operator("jg4d.vitrine_rand_dust")
        r.operator("jg4d.vitrine_rand_light")
        box.prop(p, "var_light", slider=True)

        box = L.box()
        box.label(text="Stereo", icon="CAMERA_STEREO")
        box.prop(p, "lens_mm")
        box.prop(p, "cam_distance_mm")
        box.prop(p, "target_parallax")
        box.prop(p, "reset_camera")
        box.operator("jg4d.vitrine_solve_stereo", icon="DRIVER_DISTANCE")

        box = L.box()
        box.label(text="Halftone", icon="TEXTURE")
        box.prop(p, "auto_safe_ruling")
        box.operator("jg4d.vitrine_safe_ruling", icon="CHECKMARK")
        box.prop(p, "halftone_ruling")
        box.prop(p, "halftone_angle")
        box.prop(p, "halftone_amount", slider=True)
        box.prop(p, "halftone_albedo", slider=True)


class JG4D_PT_vitrine_detail(bpy.types.Panel):
    bl_label = "Details"
    bl_parent_id = "JG4D_PT_vitrine"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "jg4d"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        p = context.scene.jg4d_vitrine
        L = self.layout

        b = L.box(); b.label(text="Item")
        b.prop(p, "override_size")
        if p.override_size:
            b.prop(p, "width_mm")
        b.prop(p, "item_depth_mm")
        b.prop(p, "item_tilt")
        b.prop(p, "curl_mm")
        b.prop(p, "paper_res")
        b.prop(p, "paper_roughness", slider=True)
        b.prop(p, "paper_bump", slider=True)

        b = L.box(); b.label(text="Case")
        r = b.row(align=True)
        r.prop(p, "case_width_mm", text="W")
        r.prop(p, "case_height_mm", text="H")
        r.prop(p, "case_depth_mm", text="D")
        b.prop(p, "glass_thickness_mm")
        b.prop(p, "pedestal_albedo", slider=True)

        b = L.box(); b.label(text="Glass")
        b.prop(p, "glass_roughness", slider=True)
        b.prop(p, "scuff_roughness", slider=True)
        b.prop(p, "scuff_amount", slider=True)
        b.prop(p, "use_scuff_map")

        b = L.box(); b.label(text="Volume + dust")
        b.prop(p, "volume_density")
        b.prop(p, "volume_anisotropy", slider=True)
        b.prop(p, "dust_count")
        b.prop(p, "dust_size_mm")
        b.prop(p, "dust_animate")
        b.prop(p, "dust_drift_mm")
        b.prop(p, "dust_speed")

        b = L.box(); b.label(text="Lights")
        r = b.row(align=True); r.prop(p, "key_energy", text="Key W"); r.prop(p, "key_kelvin", text="K")
        r = b.row(align=True); r.prop(p, "fill_energy", text="Fill W"); r.prop(p, "fill_kelvin", text="K")
        r = b.row(align=True); r.prop(p, "sheen_energy", text="Sheen W"); r.prop(p, "sheen_kelvin", text="K")
        b.prop(p, "room_hdri", text="Room")
        b.prop(p, "room_strength")
        b.prop(p, "room_rotation")


CLASSES = (
    JG4D_VitrineProps,
    JG4D_OT_vitrine_build,
    JG4D_OT_vitrine_safe_ruling,
    JG4D_OT_vitrine_stereo,
    JG4D_OT_rand_curl,
    JG4D_OT_rand_scuff,
    JG4D_OT_rand_dust,
    JG4D_OT_rand_light,
    JG4D_OT_rand_all,
    JG4D_PT_vitrine,
    JG4D_PT_vitrine_detail,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.jg4d_vitrine = bpy.props.PointerProperty(type=JG4D_VitrineProps)
    if jg4d_vitrine_frame_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(jg4d_vitrine_frame_handler)


def unregister():
    if jg4d_vitrine_frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(jg4d_vitrine_frame_handler)
    del bpy.types.Scene.jg4d_vitrine
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
