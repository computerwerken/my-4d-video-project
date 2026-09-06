# jg4d LDI3 player for Blender  (v3 — timeline-native, handler-free)
# -----------------------------------------------------------------------------
# Imports Lifecast LDI3 volumetric frames as three real displaced layer meshes,
# so Cycles/Eevee handle occlusion, lighting and stereo natively.
#
# WHY v3: v2 had no timeline support (Prev/Next buttons only) because mutating
# meshes from a frame_change handler crashed Blender 4.x. v3 needs no Python at
# playback time at all:
#   * colour + alpha  -> Image *sequences* in the material (Blender follows the
#                        scene frame by itself, viewport and render),
#   * depth           -> a Geometry Nodes modifier samples the depth cell of the
#                        same sequence (Scene Time -> frame), does the 12-bit
#                        decode with math nodes and displaces each vertex
#                        radially.  Scrubbing, playback and animation renders
#                        therefore just work; camera animation is plain Blender.
#
# Scene frame f shows file (first + f - 1); the importer sets the frame range to
# 1..count.  Decode parameters live as inputs on each layer's "jg4d_depth"
# modifier (inv_depth_coef, min/max depth...), seeded from DEFAULTS and the
# optional JSON sidecar (<file>.json or jg4d_sidecar.json next to the frames).
#
# Decode math mirrors web/lifecast_res/LifecastVideoPlayerShaders11.js.
# Grid 3x3: row 0 (bottom) = background, rows 1..2 = foreground; columns
# colour | inverse-depth | alpha; 12-bit depth = lo/hi byte pair (top half of
# the depth cell, left/right quadrants) with a fold/unfold ECC.
#
# Requires Blender 3.3+ (Geometry Nodes: Named Attribute, Scene Time,
# Separate Color).  MIT License. LDI3 format and decode math (c) Lifecast
# Incorporated (MIT).

bl_info = {
    "name": "jg4d LDI3 player",
    "author": "jg + claude",
    "version": (3, 0, 0),
    "blender": (3, 3, 0),
    "location": "3D View > Sidebar (N) > jg4d",
    "description": "Import Lifecast LDI3 frame sequences as timeline-driven displaced layer meshes",
    "category": "Import-Export",
}

import bpy
import json
import os
import re
import tempfile
import numpy as np

NUM_LAYERS = 3
GN_GROUP = "jg4d_ldi3_depth"

DEFAULTS = {
    "inv_depth_coef": 0.3,
    "ftheta_scale": 1.15,
    "ftheta_inflation": 3.0,
    "max_depth": 50.0,
    "min_depth": 0.01,
    "decode_12bit": True,
    "grid_n": 512,       # geometric parity with the reference players
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def set_noncolor(img):
    for name in ("Non-Color", "Non-Colour Data", "Raw", "Linear"):
        try:
            img.colorspace_settings.name = name
            return
        except TypeError:
            continue


def grid_dirs_uvs(grid_n, ftheta_scale, inflation):
    """Unit view directions + [0,1]^2 grid UVs for the equiangular dome."""
    n = grid_n
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    u = (i / n).ravel()
    v = (j / n).ravel()
    a = 2.0 * (u - 0.5)
    b = 2.0 * (v - 0.5)
    theta = np.arctan2(b, a)
    r = np.sqrt(a * a + b * b) / ftheta_scale
    r = 0.5 * r + 0.5 * np.power(r, inflation)
    phi = r * np.pi / 2.0
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.cos(phi),
                     np.sin(theta) * np.sin(phi)], axis=1).astype(np.float32)
    uvs = np.stack([u, v], axis=1).astype(np.float32)
    return dirs, uvs


def build_ldi_mesh(name, grid_n, ftheta_scale, inflation):
    """ONE mesh holding all three layer domes on the unit sphere (base mesh =
    view directions).  Per point: 'jg4d_uv' (grid uv) and 'jg4d_layer' (0..2);
    per face: material_index = layer.  One mesh + one modifier means each
    sequence frame is read exactly once per evaluation (three separate
    modifiers racing for the same frame occasionally got an empty buffer)."""
    n = grid_n
    dirs, uvs = grid_dirs_uvs(n, ftheta_scale, inflation)
    nv = len(dirs)
    margin = 2
    qi, qj = np.meshgrid(np.arange(n), np.arange(n))
    di = qi - n / 2.0
    dj = qj - n / 2.0
    keep = (di * di + dj * dj) <= ((n + margin) ** 2) / 4.0
    qi, qj = qi[keep], qj[keep]
    a0 = qi + (n + 1) * qj
    faces0 = np.stack([a0, a0 + (n + 1), a0 + (n + 1) + 1, a0 + 1], axis=1)
    verts = np.concatenate([dirs] * NUM_LAYERS, axis=0)
    faces = np.concatenate([faces0 + layer * nv for layer in range(NUM_LAYERS)], axis=0)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    uv_all = np.concatenate([uvs] * NUM_LAYERS, axis=0)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_verts = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    uv_layer.data.foreach_set("uv", uv_all[loop_verts].ravel())
    attr = mesh.attributes.new("jg4d_uv", "FLOAT_VECTOR", "POINT")
    uv3 = np.zeros((len(uv_all), 3), np.float32)
    uv3[:, :2] = uv_all
    attr.data.foreach_set("vector", uv3.ravel())
    lay = mesh.attributes.new("jg4d_layer", "INT", "POINT")
    lay.data.foreach_set("value", np.repeat(np.arange(NUM_LAYERS, dtype=np.int32), nv))
    mesh.polygons.foreach_set("material_index",
                              np.repeat(np.arange(NUM_LAYERS, dtype=np.int32), len(faces0)))
    mesh.validate()
    return bpy.data.objects.new(name, mesh)


# ----------------------------------------------------------------------------
# Material: colour + alpha cells, image sequence follows the scene frame
# ----------------------------------------------------------------------------

def set_image_user(iu, seq):
    """Scene frame f -> file number seq['first'] + f - 1 (frames 1..count)."""
    if not seq:
        return
    iu.frame_start = 1
    iu.frame_duration = seq["count"]
    iu.frame_offset = seq["first"] - 1
    iu.use_auto_refresh = True
    iu.use_cyclic = False


def make_material(name, img_color, img_alpha, layer, seq):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    # blend_method / shadow_method don't exist in EEVEE-Next (4.2+); guard them.
    for attr, val in (("blend_method", "HASHED"), ("shadow_method", "HASHED")):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    nt = mat.node_tree
    nt.nodes.clear()

    def node(t, x, y):
        nn = nt.nodes.new(t); nn.location = (x, y); return nn

    uvmap = node("ShaderNodeUVMap", -900, 0); uvmap.uv_map = "UVMap"
    map_c = node("ShaderNodeMapping", -700, 100)
    map_c.inputs["Scale"].default_value = (1 / 3, 1 / 3, 1)
    map_c.inputs["Location"].default_value = (0.0, layer / 3.0, 0)
    map_a = node("ShaderNodeMapping", -700, -250)
    map_a.inputs["Scale"].default_value = (1 / 3, 1 / 3, 1)
    map_a.inputs["Location"].default_value = (2 / 3, layer / 3.0, 0)
    tex_c = node("ShaderNodeTexImage", -450, 100); tex_c.image = img_color; tex_c.extension = "EXTEND"
    tex_a = node("ShaderNodeTexImage", -450, -250); tex_a.image = img_alpha; tex_a.extension = "EXTEND"
    set_image_user(tex_c.image_user, seq)
    set_image_user(tex_a.image_user, seq)
    emit = node("ShaderNodeEmission", -150, 100)
    transp = node("ShaderNodeBsdfTransparent", -150, -100)
    mix = node("ShaderNodeMixShader", 50, 0)
    out = node("ShaderNodeOutputMaterial", 250, 0)
    L = nt.links.new
    L(uvmap.outputs["UV"], map_c.inputs["Vector"])
    L(uvmap.outputs["UV"], map_a.inputs["Vector"])
    L(map_c.outputs["Vector"], tex_c.inputs["Vector"])
    L(map_a.outputs["Vector"], tex_a.inputs["Vector"])
    L(tex_c.outputs["Color"], emit.inputs["Color"])
    L(tex_a.outputs["Color"], mix.inputs["Fac"])
    L(transp.outputs["BSDF"], mix.inputs[1])
    L(emit.outputs["Emission"], mix.inputs[2])
    L(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


# ----------------------------------------------------------------------------
# Geometry Nodes: depth decode + radial displacement (this replaces the numpy
# decode of v1/v2 and is what makes the timeline work)
# ----------------------------------------------------------------------------

def _enabled(sockets, name=None):
    """First enabled socket (optionally by name). Blender 3.x nodes such as
    Switch / Named Attribute carry one hidden socket per data type."""
    for s in sockets:
        if s.enabled and (name is None or s.name == name):
            return s
    return sockets[0]


def _group_socket(ng, name, stype, in_out, default=None):
    """Blender 4.0+ (ng.interface) and 3.x (ng.inputs/outputs) compatible."""
    if hasattr(ng, "interface"):
        s = ng.interface.new_socket(name=name, in_out=in_out, socket_type=stype)
    else:
        s = (ng.inputs if in_out == "INPUT" else ng.outputs).new(stype, name)
    if default is not None:
        s.default_value = default
    return s.identifier


def get_or_build_depth_group():
    ng = bpy.data.node_groups.get(GN_GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(GN_GROUP, "GeometryNodeTree")
    ids = {}
    _group_socket(ng, "Geometry", "NodeSocketGeometry", "INPUT")
    _group_socket(ng, "Geometry", "NodeSocketGeometry", "OUTPUT")
    ids["image"] = _group_socket(ng, "image", "NodeSocketImage", "INPUT")
    ids["frame_offset"] = _group_socket(ng, "frame_offset", "NodeSocketInt", "INPUT", 0)
    ids["inv_depth_coef"] = _group_socket(ng, "inv_depth_coef", "NodeSocketFloat", "INPUT", 0.3)
    ids["min_depth"] = _group_socket(ng, "min_depth", "NodeSocketFloat", "INPUT", 0.01)
    ids["max_depth"] = _group_socket(ng, "max_depth", "NodeSocketFloat", "INPUT", 50.0)
    ids["twelve_bit"] = _group_socket(ng, "twelve_bit", "NodeSocketBool", "INPUT", True)
    for k in range(NUM_LAYERS):
        ids["show_layer%d" % k] = _group_socket(ng, "show_layer%d" % k, "NodeSocketBool", "INPUT", True)
    ng["jg4d_ids"] = json.dumps(ids)

    nodes, links = ng.nodes, ng.links
    L = links.new

    def node(t, x, y, **kw):
        nn = nodes.new(t); nn.location = (x, y)
        for k, v in kw.items():
            setattr(nn, k, v)
        return nn

    def math(op, x, y, a=None, b=None, c=None, label=None):
        nn = node("ShaderNodeMath", x, y, operation=op)
        if label:
            nn.label = label
        for i, val in enumerate((a, b, c)):
            if val is None:
                continue
            if isinstance(val, bpy.types.NodeSocket):
                L(val, nn.inputs[i])
            else:
                nn.inputs[i].default_value = val
        return nn.outputs[0]

    gi = node("NodeGroupInput", -1400, 0)
    go = node("NodeGroupOutput", 1400, 0)

    # grid uv (stored per point at import) -> u, v  (clamped just below 1 so the
    # last row/column never samples the neighbouring quadrant/cell)
    uv = node("GeometryNodeInputNamedAttribute", -1400, -400, data_type="FLOAT_VECTOR")
    uv.inputs["Name"].default_value = "jg4d_uv"
    sep = node("ShaderNodeSeparateXYZ", -1200, -400)
    L(_enabled(uv.outputs), sep.inputs["Vector"])
    u = math("MINIMUM", -1100, -350, sep.outputs["X"], 0.99999)
    v = math("MINIMUM", -1100, -450, sep.outputs["Y"], 0.99999)

    # layer index (stored per point at import) -> cell row offset
    lay = node("GeometryNodeInputNamedAttribute", -1400, -700, data_type="INT")
    lay.inputs["Name"].default_value = "jg4d_layer"
    layer = _enabled(lay.outputs)
    cell_v = math("MULTIPLY", -1200, -650, layer, 1.0 / 3.0, label="layer/3")

    # frame for the sequence: scene frame + offset  (file number)
    st = node("GeometryNodeInputSceneTime", -1400, 300)
    frame = math("ADD", -1200, 300, st.outputs["Frame"], gi.outputs["frame_offset"], label="file frame")

    def sample(x, y, uu, vv, label):
        """Red channel of the depth image at image-space (uu, vv)."""
        comb = node("ShaderNodeCombineXYZ", x, y)
        L(uu, comb.inputs["X"]); L(vv, comb.inputs["Y"])
        tex = node("GeometryNodeImageTexture", x + 200, y, interpolation="Closest", extension="EXTEND")
        tex.label = label
        L(gi.outputs["image"], tex.inputs["Image"])
        L(comb.outputs["Vector"], tex.inputs["Vector"])
        L(frame, tex.inputs["Frame"])
        sc = node("FunctionNodeSeparateColor", x + 500, y)
        L(tex.outputs["Color"], sc.inputs["Color"])
        return sc.outputs["Red"]

    # --- 12-bit path: lo at (1/3 + u/6, cell_v + (v+1)/6), hi at (1/3 + (u+1)/6, same v)
    u6 = math("MULTIPLY", -1000, -300, u, 1.0 / 6.0)
    lo_x = math("ADD", -800, -250, u6, 1.0 / 3.0)
    hi_x = math("ADD", -800, -350, u6, 0.5)
    v6 = math("MULTIPLY_ADD", -1000, -500, v, 1.0 / 6.0, 1.0 / 6.0)      # (v+1)/6
    top_y = math("ADD", -800, -500, v6, cell_v)
    lo = sample(-600, -200, lo_x, top_y, "depth lo")
    hi = sample(-600, -500, hi_x, top_y, "depth hi")

    lo_b = math("ROUND", 100, -200, math("MULTIPLY", -50, -200, lo, 255.0))
    hi_b = math("ROUND", 100, -500, math("MULTIPLY", -50, -500, hi, 255.0))
    hi4 = math("FLOOR", 400, -500, math("DIVIDE", 250, -500, hi_b, 16.0), label="hi nibble")
    parity = math("MODULO", 550, -650, hi4, 2.0, label="fold parity")
    # lo' = lo + parity * (255 - 2*lo)
    t = math("MULTIPLY_ADD", 250, -300, lo_b, -2.0, 255.0)
    lo_adj = math("MULTIPLY_ADD", 550, -300, parity, t, lo_b, label="unfold")
    i12 = math("MULTIPLY_ADD", 750, -400, hi4, 256.0, lo_adj, label="12-bit value")
    invd12 = math("DIVIDE", 900, -400, i12, 4095.0)

    # --- 8-bit path: plain depth cell at (1/3 + u/3, cell_v + v/3)
    x8 = math("MULTIPLY_ADD", -1000, 100, u, 1.0 / 3.0, 1.0 / 3.0)
    y8 = math("MULTIPLY_ADD", -1000, 0, v, 1.0 / 3.0, cell_v)
    invd8 = sample(-600, 100, x8, y8, "depth 8-bit")

    sw = node("GeometryNodeSwitch", 1050, -200, input_type="FLOAT")
    L(gi.outputs["twelve_bit"], _enabled(sw.inputs, "Switch"))
    L(invd8, _enabled(sw.inputs, "False"))
    L(invd12, _enabled(sw.inputs, "True"))
    invd = math("MAXIMUM", 1100, -400, _enabled(sw.outputs), 1e-4)

    # radius = clamp(coef / invd, min_depth, max_depth); position = dir * radius
    radius = node("ShaderNodeClamp", 1100, -600)
    L(math("DIVIDE", 950, -600, gi.outputs["inv_depth_coef"], invd), radius.inputs["Value"])
    L(gi.outputs["min_depth"], radius.inputs["Min"])
    L(gi.outputs["max_depth"], radius.inputs["Max"])
    pos = node("GeometryNodeInputPosition", 950, 200)
    scale = node("ShaderNodeVectorMath", 1100, 100, operation="SCALE")
    L(pos.outputs["Position"], scale.inputs[0])
    L(radius.outputs["Result"], scale.inputs["Scale"])
    setp = node("GeometryNodeSetPosition", 1250, 0)
    L(gi.outputs["Geometry"], setp.inputs["Geometry"])
    L(scale.outputs["Vector"], setp.inputs["Position"])

    # layer visibility: delete the points of any layer whose show_layerN is off
    hide_any = None
    for k in range(NUM_LAYERS):
        is_k = node("FunctionNodeCompare", 1250, -300 - 150 * k, data_type="INT", operation="EQUAL")
        a_in, b_in = [s for s in is_k.inputs if s.enabled][:2]              # int A, int B
        L(layer, a_in); b_in.default_value = k
        hidden = node("FunctionNodeBooleanMath", 1400, -300 - 150 * k, operation="NOT")
        L(gi.outputs["show_layer%d" % k], hidden.inputs[0])
        both = node("FunctionNodeBooleanMath", 1550, -300 - 150 * k, operation="AND")
        L(is_k.outputs["Result"], both.inputs[0]); L(hidden.outputs["Boolean"], both.inputs[1])
        if hide_any is None:
            hide_any = both.outputs["Boolean"]
        else:
            acc = node("FunctionNodeBooleanMath", 1700, -300 - 150 * k, operation="OR")
            L(hide_any, acc.inputs[0]); L(both.outputs["Boolean"], acc.inputs[1])
            hide_any = acc.outputs["Boolean"]
    delete = node("GeometryNodeDeleteGeometry", 1850, 0, domain="POINT")
    L(setp.outputs["Geometry"], delete.inputs["Geometry"])
    L(hide_any, delete.inputs["Selection"])
    go.location = (2050, 0)
    L(delete.outputs["Geometry"], go.inputs["Geometry"])
    return ng


def add_depth_modifier(obj, img_depth, p, seq):
    ng = get_or_build_depth_group()
    ids = json.loads(ng["jg4d_ids"])
    mod = obj.modifiers.new("jg4d_depth", "NODES")
    mod.node_group = ng
    mod[ids["image"]] = img_depth
    mod[ids["frame_offset"]] = (seq["first"] - 1) if seq else 0
    mod[ids["inv_depth_coef"]] = float(p["inv_depth_coef"])
    mod[ids["min_depth"]] = float(p["min_depth"])
    mod[ids["max_depth"]] = float(p["max_depth"])
    mod[ids["twelve_bit"]] = bool(p["decode_12bit"])
    for k in range(NUM_LAYERS):
        mod[ids["show_layer%d" % k]] = True
    return mod


# ----------------------------------------------------------------------------
# Sequence + sidecar
# ----------------------------------------------------------------------------

def parse_sequence(filepath):
    d, base = os.path.split(filepath)
    m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", base)
    if not m:
        return None
    stem, digits, ext = m.groups()
    rx = re.compile(r"^%s(\d+)%s$" % (re.escape(stem), re.escape(ext)))
    frames = sorted(int(rx.match(f).group(1)) for f in os.listdir(d) if rx.match(f))
    if not frames:
        return None
    return {"first": frames[0], "last": frames[-1], "count": len(frames),
            "contiguous": frames[-1] - frames[0] + 1 == len(frames),
            "first_path": os.path.join(d, "%s%0*d%s" % (stem, len(digits), frames[0], ext))}


def load_sidecar(filepath):
    for cand in (os.path.splitext(filepath)[0] + ".json",
                 os.path.join(os.path.dirname(filepath), "jg4d_sidecar.json")):
        if os.path.isfile(cand):
            try:
                data = json.load(open(cand))
                return {k: data[k] for k in DEFAULTS if k in data}
            except Exception as e:
                print("jg4d: sidecar read failed:", e)
    return {}


# ----------------------------------------------------------------------------
# Build the rig
# ----------------------------------------------------------------------------

def load_sequence_image(path, seq):
    img = bpy.data.images.load(path, check_existing=False)
    if seq and seq["count"] > 1:
        img.source = "SEQUENCE"
    return img


def build_rig(context, filepath):
    p = dict(DEFAULTS); p.update(load_sidecar(filepath))
    seq = parse_sequence(filepath)
    if seq and seq["count"] < 2:
        seq = None
    first_path = seq["first_path"] if seq else filepath

    root = bpy.data.objects.new("jg4d_ldi3", None)
    root.empty_display_size = 0.2
    context.scene.collection.objects.link(root)

    img_c = load_sequence_image(first_path, seq)            # colour (sRGB)
    img_d = load_sequence_image(first_path, seq)            # alpha + depth (raw)
    set_noncolor(img_d)

    obj = build_ldi_mesh("jg4d_ldi3_mesh", p["grid_n"],
                         p["ftheta_scale"], p["ftheta_inflation"])
    context.scene.collection.objects.link(obj)
    obj.parent = root
    for layer in range(NUM_LAYERS):   # slot order must match material_index = layer
        obj.data.materials.append(make_material("jg4d_layer%d" % layer, img_c, img_d, layer, seq))
    add_depth_modifier(obj, img_d, p, seq)

    if seq:
        s = context.scene
        s.frame_start = 1
        s.frame_end = seq["count"]
        s.frame_current = 1

    root["jg4d"] = json.dumps({
        "params": p, "filepath": first_path, "img_c": img_c.name, "img_d": img_d.name,
        "mesh": obj.name, "nv": (p["grid_n"] + 1) ** 2,
        "first": seq["first"] if seq else 0,
        "last": seq["last"] if seq else 0,
        "count": seq["count"] if seq else 1,
        "contiguous": seq["contiguous"] if seq else True,
    })
    return root


def find_root(context):
    for obj in context.scene.collection.all_objects:
        if obj.type == "EMPTY" and "jg4d" in obj:
            return obj
    return None


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------

class JG4D_OT_import(bpy.types.Operator):
    bl_idname = "jg4d.import_ldi3"
    bl_label = "Import LDI3 sequence"
    bl_options = {"REGISTER", "UNDO"}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg", options={"HIDDEN"})

    def execute(self, context):
        try:
            root = build_rig(context, self.filepath)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.report({"ERROR"}, "jg4d import failed: %s" % e)
            return {"CANCELLED"}
        cfg = json.loads(root["jg4d"])
        msg = "Imported %d frame(s): files %d..%d -> scene frames 1..%d" % (
            cfg["count"], cfg["first"], cfg["last"], cfg["count"])
        if not cfg["contiguous"]:
            msg += "  (WARNING: numbering has gaps; missing files show as holes)"
        self.report({"INFO"}, msg)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class JG4D_OT_frame_range(bpy.types.Operator):
    bl_idname = "jg4d.frame_range"
    bl_label = "Reset frame range to sequence"

    def execute(self, context):
        root = find_root(context)
        if not root:
            self.report({"ERROR"}, "No jg4d import in scene"); return {"CANCELLED"}
        cfg = json.loads(root["jg4d"])
        context.scene.frame_start = 1
        context.scene.frame_end = cfg["count"]
        return {"FINISHED"}


class JG4D_OT_stereo(bpy.types.Operator):
    bl_idname = "jg4d.setup_stereo"
    bl_label = "Setup stereo camera (anaglyph)"

    def execute(self, context):
        s = context.scene
        s.render.use_multiview = True
        s.render.views_format = "STEREO_3D"
        s.render.image_settings.views_format = "STEREO_3D"
        s.render.image_settings.stereo_3d_format.display_mode = "ANAGLYPH"
        s.render.image_settings.stereo_3d_format.anaglyph_type = "RED_CYAN"
        cam = s.camera
        if cam is None:
            cam = bpy.data.objects.new("jg4d_cam", bpy.data.cameras.new("jg4d_cam"))
            s.collection.objects.link(cam); s.camera = cam
        cam.location = (0, 0, 0)
        cam.rotation_euler = (1.5708, 0, 0)   # look down +Y (capture forward)
        cam.data.stereo.interocular_distance = 0.063
        cam.data.stereo.convergence_mode = "OFFAXIS"
        cam.data.stereo.convergence_distance = 2.0
        self.report({"INFO"}, "Stereo on. Viewport: View > Stereoscopy. Camera view = camera icon in the gizmo.")
        return {"FINISHED"}


DUBOIS_L = np.array([[0.4561, 0.500484, 0.176381],
                     [-0.0400822, -0.0378246, -0.0157589],
                     [-0.0152161, -0.0205971, -0.0054686]], np.float32)
DUBOIS_R = np.array([[-0.0434706, -0.0879388, -0.0015553],
                     [0.378476, 0.73364, -0.0180517],
                     [-0.0721527, -0.112961, 1.2264]], np.float32)


class JG4D_OT_dubois(bpy.types.Operator):
    bl_idname = "jg4d.dubois"
    bl_label = "Dubois anaglyph from L/R pair"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.tif;*.tiff", options={"HIDDEN"})

    def _read(self, path):
        img = bpy.data.images.load(path, check_existing=False)
        try:
            set_noncolor(img)
            n = len(img.pixels); w, h = img.size; ch = n // (w * h)
            buf = np.empty(n, np.float32); img.pixels.foreach_get(buf)
            return buf.reshape(h, w, ch)[:, :, :3][::-1].copy()
        finally:
            bpy.data.images.remove(img)

    def execute(self, context):
        lp = self.filepath
        rp = None
        for lt, rt in (("_L", "_R"), ("left", "right"), ("_l", "_r")):
            if lt in os.path.basename(lp):
                cand = os.path.join(os.path.dirname(lp), os.path.basename(lp).replace(lt, rt))
                if os.path.isfile(cand):
                    rp = cand; ltag = lt; break
        if not rp:
            self.report({"ERROR"}, "No matching right-eye file"); return {"CANCELLED"}
        try:
            l = np.power(np.clip(self._read(lp), 0, 1), 2.2)
            r = np.power(np.clip(self._read(rp), 0, 1), 2.2)
            out = np.power(np.clip(l @ DUBOIS_L.T + r @ DUBOIS_R.T, 0, 1), 1 / 2.2)
            h, w = out.shape[:2]
            oi = bpy.data.images.new("__jg4d_ana", w, h, alpha=False)
            rgba = np.ones((h, w, 4), np.float32); rgba[:, :, :3] = out[::-1]
            oi.pixels.foreach_set(rgba.ravel())
            op = os.path.splitext(lp)[0].replace(ltag, "") + "_anaglyph.png"
            oi.filepath_raw = op; oi.file_format = "PNG"; oi.save()
            bpy.data.images.remove(oi)
        except Exception as e:
            self.report({"ERROR"}, "Dubois failed: %s" % e); return {"CANCELLED"}
        self.report({"INFO"}, "Wrote %s" % op)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


# ----------------------------------------------------------------------------
# Self-test: synthetic 2-frame LDI3 sequence with known depths -> import ->
# evaluate the modifier at frames 1 and 2 -> compare vertex radii.
# ----------------------------------------------------------------------------

def _encode12(invd):
    i12 = int(round(float(np.clip(invd, 0, 1)) * 4095))
    hi4, lo = i12 >> 8, i12 & 255
    if hi4 % 2 == 1:
        lo = 255 - lo
    return lo, hi4 * 17        # hi byte: nibble duplicated, decode uses hi // 16


def _write_synthetic_frame(path, cell, values):
    """values[layer] = (invd_left, invd_right) for u<0.5 / u>=0.5.  Writes a
    3x3 LDI3 grid PNG through Blender's own image writer (Non-Color = raw bytes)."""
    W = 3 * cell
    px = np.zeros((W, W, 4), np.float32)      # row 0 = image BOTTOM (Blender order)
    px[:, :, 3] = 1.0
    half = cell // 2          # quadrant size
    q = half // 2             # half a quadrant = the u<0.5 / u>=0.5 split
    for layer in range(NUM_LAYERS):
        y0 = layer * cell
        px[y0:y0 + cell, 0:cell, 0] = 0.25 + 0.25 * layer          # colour cell (flat)
        px[y0:y0 + cell, 2 * cell:3 * cell, :3] = 1.0              # alpha cell = opaque
        rows = slice(y0 + half, y0 + cell)                         # top half of the depth cell
        for side, invd in enumerate(values[layer]):
            lo, hi = _encode12(invd)
            lo_x0 = cell + side * q                                # lo quadrant: x in [cell, cell+half)
            hi_x0 = cell + half + side * q                         # hi quadrant: x in [cell+half, 2cell)
            px[rows, lo_x0:lo_x0 + q, :3] = lo / 255.0
            px[rows, hi_x0:hi_x0 + q, :3] = hi / 255.0
    img = bpy.data.images.new("__jg4d_synth", W, W, alpha=True)
    set_noncolor(img)
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


class JG4D_OT_selftest(bpy.types.Operator):
    bl_idname = "jg4d.selftest"
    bl_label = "Self-test decode + timeline"
    bl_description = "Builds a tiny synthetic LDI3 sequence with known depths, imports it and checks the displaced geometry at frames 1 and 2"

    def execute(self, context):
        d = tempfile.mkdtemp(prefix="jg4d_selftest_")
        # (layer) -> (invd for u<0.5, invd for u>=0.5); frame 2 = frame 1 halved
        f1 = [(0.30, 0.15), (0.60, 0.40), (1.00, 0.75)]
        f2 = [(a / 2, b / 2) for a, b in f1]
        try:
            _write_synthetic_frame(os.path.join(d, "ldi3_000007.png"), 48, f1)
            _write_synthetic_frame(os.path.join(d, "ldi3_000008.png"), 48, f2)
            json.dump({"grid_n": 16}, open(os.path.join(d, "jg4d_sidecar.json"), "w"))
            root = build_rig(context, os.path.join(d, "ldi3_000007.png"))
            cfg = json.loads(root["jg4d"])
            coef = cfg["params"]["inv_depth_coef"]
            dirs, uvs = grid_dirs_uvs(16, cfg["params"]["ftheta_scale"], cfg["params"]["ftheta_inflation"])
            nv = cfg["nv"]
            mesh_obj = bpy.data.objects[cfg["mesh"]]
            worst = 0.0
            for frame, vals in ((1, f1), (2, f2)):
                context.scene.frame_set(frame)
                dg = context.evaluated_depsgraph_get()
                ev = mesh_obj.evaluated_get(dg)
                co = np.empty(len(ev.data.vertices) * 3, np.float32)
                ev.data.vertices.foreach_get("co", co)
                radii = np.linalg.norm(co.reshape(-1, 3), axis=1)
                if len(radii) != nv * NUM_LAYERS:
                    raise RuntimeError("evaluated mesh has %d verts, expected %d"
                                       % (len(radii), nv * NUM_LAYERS))
                for layer in range(NUM_LAYERS):
                    rad = radii[layer * nv:(layer + 1) * nv]   # layers are stacked in order
                    # exact quantised expectation, same rounding as the encoder
                    exp = np.where(uvs[:, 0] < 0.5,
                                   coef / (round(vals[layer][0] * 4095) / 4095.0),
                                   coef / (round(vals[layer][1] * 4095) / 4095.0))
                    inner = uvs[:, 0] != 0.5          # skip the seam column
                    err = np.abs(rad - exp)[inner].max()
                    worst = max(worst, float(err))
                    print("jg4d selftest frame %d layer %d: max radius err %.5f (expect %.3f / %.3f)"
                          % (frame, layer, err, exp.min(), exp.max()))
            # tidy up
            for child in list(root.children):
                bpy.data.objects.remove(child, do_unlink=True)
            bpy.data.objects.remove(root, do_unlink=True)
            for nm in (cfg["img_c"], cfg["img_d"]):
                im = bpy.data.images.get(nm)
                if im:
                    bpy.data.images.remove(im)
            import shutil; shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.report({"ERROR"}, "self-test crashed: %s" % e)
            return {"CANCELLED"}
        if worst < 1e-3:
            msg = "PASS: decode + frame switching OK (max err %.2e)" % worst
            print("jg4d selftest", msg); self.report({"INFO"}, msg)
        else:
            msg = "FAIL: max radius error %.4f (see console)" % worst
            print("jg4d selftest", msg); self.report({"ERROR"}, msg)
        return {"FINISHED"}


class JG4D_PT_panel(bpy.types.Panel):
    bl_label = "jg4d LDI3 player"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "jg4d"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.operator("jg4d.import_ldi3", icon="IMPORT")
        root = find_root(context)
        if root:
            cfg = json.loads(root["jg4d"])
            box = self.layout.box()
            box.label(text="files %d..%d = scene frames 1..%d" % (cfg["first"], cfg["last"], cfg["count"]))
            box.label(text="Scrub / play the timeline.", icon="TIME")
            box.operator("jg4d.frame_range", icon="PREVIEW_RANGE")
            mesh_obj = bpy.data.objects.get(cfg.get("mesh", ""))
            mod = mesh_obj.modifiers.get("jg4d_depth") if mesh_obj else None
            if mod and mod.node_group and "jg4d_ids" in mod.node_group:
                ids = json.loads(mod.node_group["jg4d_ids"])
                row = box.row(align=True)
                row.label(text="Layers:")
                for k in range(NUM_LAYERS):
                    row.prop(mod, '["%s"]' % ids["show_layer%d" % k], text=str(k), toggle=True)
                for key in ("inv_depth_coef", "min_depth", "max_depth"):
                    box.prop(mod, '["%s"]' % ids[key], text=key)
            else:
                box.label(text="Depth params: modifier 'jg4d_depth' on the mesh")
        self.layout.separator()
        c2 = self.layout.column(align=True)
        c2.operator("jg4d.setup_stereo", icon="CAMERA_STEREO")
        c2.operator("jg4d.dubois", icon="IMAGE_RGB")
        self.layout.separator()
        self.layout.operator("jg4d.selftest", icon="CHECKMARK")


CLASSES = (JG4D_OT_import, JG4D_OT_frame_range, JG4D_OT_stereo, JG4D_OT_dubois,
           JG4D_OT_selftest, JG4D_PT_panel)


def _purge_jg4d_handlers():
    """Remove stray handlers left by v1 (its persistent frame handler is the
    crash source). Safe to call repeatedly."""
    for lst in (bpy.app.handlers.frame_change_post,
                bpy.app.handlers.frame_change_pre,
                bpy.app.handlers.depsgraph_update_post):
        for h in list(lst):
            if "jg4d" in getattr(h, "__name__", ""):
                try:
                    lst.remove(h)
                except ValueError:
                    pass


def register():
    _purge_jg4d_handlers()
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    _purge_jg4d_handlers()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
