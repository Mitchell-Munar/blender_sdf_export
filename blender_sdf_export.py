"""
Blender -> Gazebo (gz sim) world exporter
=========================================

Exports a Blender collection as a ready-to-run Gazebo world:

    <output_dir>/
        worlds/<world_name>.sdf          world file (create or update)
        models/<model>/model.sdf         one model per unique mesh (or per merged collection)
        models/<model>/model.config
        models/<model>/meshes/<model>.obj|.glb (+ .mtl and textures for OBJ)
        run_world.sh                     helper that sets the resource path and launches Gazebo

HOW TO RUN
----------
A) Inside Blender (Scripting tab -> open this file -> Run Script):
   a folder-picker dialog opens so you choose where everything is saved.
   After the first run it is also available under File > Export > Gazebo World.

B) Headless / command line:
   blender -b my_scene.blend --python export_gazebo_world.py -- \
       --output-dir ~/ROS2/swerve_bot/src/swervebot_sim \
       --collection GazeboExport --merge Walls,Furniture

PERFORMANCE: "merged collections"
---------------------------------
Every collection listed in `merged_collections` is baked into ONE mesh / ONE model
(one visual + one collision) instead of one model per object. Fewer entities means
far less overhead for Gazebo. Materials are preserved as separate sub-meshes.
Unlisted collections export one model per object; objects sharing the same mesh
datablock (and scale) share one mesh file, so repeated objects cost almost nothing.

UPDATING
--------
Re-running is safe:
  * Models written by this script are overwritten; models that no longer exist in
    Blender are removed (only folders this script created, tracked by a marker file).
  * If the world file already exists and contains the BLENDER_EXPORT markers, only the
    block between the markers is replaced, so your hand-written additions (robot,
    plugins, sensors) outside the markers are kept. A .bak copy is saved first.

PER-OBJECT / PER-COLLECTION CUSTOM PROPERTIES (Object or Collection properties panel)
    gz_collision = "mesh" | "box" | "auto" | "none"   override the collision type
    gz_ignore    = 1                         skip this object (objects only)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import bmesh
import bpy
import numpy as np
import math
from mathutils import Euler, Matrix, Vector

MARKER_BEGIN = "<!-- BLENDER_EXPORT_BEGIN -->"
MARKER_END = "<!-- BLENDER_EXPORT_END -->"
MODEL_MARKER = ".blender_export"
EXPORTABLE_TYPES = {"MESH", "CURVE", "SURFACE", "FONT"}


# ----------------------------------------------------------------------------
# SETTINGS  (edit here, or override via the dialog / command-line arguments)
# ----------------------------------------------------------------------------
@dataclass
class Settings:
    output_dir: str = "~/ROS2/swerve_bot/src/swervebot_sim"
    world_name: str = "room_layout"
    collection: str = "GazeboExport"           # collection to export (children included)
    merged_collections: list = field(default_factory=list)  # e.g. ["Walls", "Furniture"]
    mesh_format: str = "obj"                   # "obj" (safest) or "glb"
    collision: str = "mesh"                    # "mesh" (exact, default) | "box" | "auto" (best-fit box/cylinder/sphere) | "none"
    gz_flavor: str = "gz"                      # "gz" (Garden/Harmonic/Ionic) | "ignition" (Fortress)
    sdf_version: str = "1.9"                   # 1.9 = Garden/Harmonic, 1.8 = Fortress
    group_identical_geometry: bool = True      # same-looking objects share ONE model (Cube, Cube.001, ...)
    strip_numeric_suffix: bool = True          # name shared models "wall_segment" instead of "wall_segment_1"
    include_ground_plane: bool = True
    include_hidden: bool = False
    remove_stale: bool = True
    backup_world: bool = True

    def finalize(self):
        if self.gz_flavor == "ignition" and self.sdf_version == "1.9":
            self.sdf_version = "1.8"
        if self.mesh_format not in ("obj", "glb"):
            raise ValueError("mesh_format must be 'obj' or 'glb'")
        if self.collision not in ("mesh", "box", "auto", "none"):
            raise ValueError("collision must be 'mesh', 'box', 'auto' or 'none'")
        if self.gz_flavor not in ("gz", "ignition"):
            raise ValueError("gz_flavor must be 'gz' or 'ignition'")
        self.world_name = sanitize(self.world_name)
        return self


SETTINGS = Settings(
    output_dir="~/ROS2/swerve_bot/src/swervebot_sim",
    world_name="room_layout",
    collection="GazeboExport",
    merged_collections=[],          # <- put collection names here to merge them into one mesh
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def sanitize(name):
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name)).strip("_")
    return s or "unnamed"


def unique_name(base, used):
    name, i = base, 1
    while name in used:
        i += 1
        name = f"{base}_{i}"
    used.add(name)
    return name


def fmt_pose(loc, euler):
    return " ".join(f"{v:.6f}" for v in (loc.x, loc.y, loc.z, euler.x, euler.y, euler.z))


def is_exportable(obj, s):
    if obj.type not in EXPORTABLE_TYPES or obj.get("gz_ignore"):
        return False
    if not s.include_hidden:
        try:
            if obj.hide_get() or obj.hide_viewport:
                return False
        except RuntimeError:
            pass
    return True


def gather(coll, s, merged, singles, groups):
    """Walk the collection tree. Collections named in `merged` become one group."""
    if coll.name in merged:
        objs = [o for o in coll.all_objects if is_exportable(o, s)]
        if objs:
            groups[coll.name] = (coll, objs)
        return
    singles.extend(o for o in coll.objects if is_exportable(o, s))
    for child in coll.children:
        gather(child, s, merged, singles, groups)


def ensure_object_mode():
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


# ----------------------------------------------------------------------------
# Mesh building
# ----------------------------------------------------------------------------
def evaluated_mesh(obj, depsgraph):
    """Mesh copy with modifiers applied (non-destructive)."""
    return bpy.data.meshes.new_from_object(
        obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph
    )


def read_material_indices(me):
    idx = np.zeros(len(me.polygons), dtype=np.int32)
    if len(idx):
        me.polygons.foreach_get("material_index", idx)
    return idx


def apply_object_materials(me, obj):
    """Make sure the mesh carries the object's actual material slots."""
    idx = read_material_indices(me)
    mats = [sl.material for sl in obj.material_slots]
    me.materials.clear()
    for m in mats:
        me.materials.append(m)
    if len(idx):
        me.polygons.foreach_set("material_index", idx)


def transform_mesh(me, matrix):
    me.transform(matrix)
    if matrix.determinant() < 0:
        me.flip_normals()
    me.update()


def build_merged_mesh(objs, depsgraph, name):
    """Join many objects into a single world-space mesh, remapping materials."""
    materials, mat_lookup = [], {}
    bm = bmesh.new()
    for obj in objs:
        me = evaluated_mesh(obj, depsgraph)
        if me is None or len(me.polygons) == 0:
            if me is not None:
                bpy.data.meshes.remove(me)
            continue
        slot_mats = [sl.material for sl in obj.material_slots]
        old_idx = read_material_indices(me)
        new_idx = np.zeros_like(old_idx)
        for i, mi in enumerate(old_idx):
            mat = slot_mats[mi] if mi < len(slot_mats) else None
            key = mat.name_full if mat else None
            if key not in mat_lookup:
                mat_lookup[key] = len(materials)
                materials.append(mat)
            new_idx[i] = mat_lookup[key]
        me.polygons.foreach_set("material_index", new_idx)

        # keep a single, consistently named UV layer so layers line up when joined
        active_uv = me.uv_layers.active
        if active_uv:
            keep = active_uv.name
            for uv in list(me.uv_layers):
                if uv.name != keep:
                    me.uv_layers.remove(uv)
            me.uv_layers[0].name = "UVMap"

        transform_mesh(me, obj.matrix_world)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)

    merged = bpy.data.meshes.new(name)
    bm.to_mesh(merged)
    bm.free()
    for mat in materials:
        merged.materials.append(mat)
    merged.update()
    return merged


def mesh_coords(me):
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def fit_shape(co, auto):
    """Fit a collision primitive to vertex coordinates `co` (N x 3, model frame).

    Always returns a box unless `auto` is on AND a cylinder/sphere that encloses
    every vertex is clearly smaller (>10 %) than the bounding box. Any problem
    (bad data, degenerate size, exception) falls back to the box.
    """
    mn, mx = co.min(axis=0), co.max(axis=0)
    center = (mn + mx) / 2.0
    size = np.maximum(mx - mn, 1e-3)
    box = {"type": "box", "center": tuple(center), "rpy": (0.0, 0.0, 0.0), "size": tuple(size)}
    if not auto or len(co) < 8 or not np.isfinite(co).all():
        return box
    try:
        half_pi = math.pi / 2.0
        candidates = []
        r = float(np.linalg.norm(co - center, axis=1).max())
        if r > 1e-4:
            candidates.append((4.0 / 3.0 * math.pi * r ** 3,
                               {"type": "sphere", "center": tuple(center), "rpy": (0.0, 0.0, 0.0), "radius": r}))
        # SDF cylinders stand along their local Z, so tilt the frame for X / Y axes
        for axis, (u, v), rpy in ((0, (1, 2), (0.0, half_pi, 0.0)),
                                  (1, (0, 2), (half_pi, 0.0, 0.0)),
                                  (2, (0, 1), (0.0, 0.0, 0.0))):
            rad = float(np.hypot(co[:, u] - center[u], co[:, v] - center[v]).max())
            length = float(size[axis])
            if rad > 1e-4:
                candidates.append((math.pi * rad * rad * length,
                                   {"type": "cylinder", "center": tuple(center), "rpy": rpy,
                                    "radius": rad, "length": length}))
        if candidates:
            vol, shape = min(candidates, key=lambda c: c[0])
            if vol < 0.9 * float(np.prod(size)):
                return shape
    except Exception as exc:  # never let fitting break an export
        print(f"[warn] shape fitting failed ({exc}); using box")
    return box


def object_shapes(objs, depsgraph, auto):
    """One fitted primitive per source object, expressed in WORLD space.
    Used for merged collections so collision does not become one giant shape."""
    shapes = []
    for obj in objs:
        me = evaluated_mesh(obj, depsgraph)
        try:
            if me is None or len(me.vertices) == 0:
                continue
            loc, rot, scale = obj.matrix_world.decompose()
            co = mesh_coords(me) * np.array([scale.x, scale.y, scale.z])
            sh = fit_shape(co, auto)
            frame = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
            local = (Matrix.Translation(Vector(sh["center"]))
                     @ Euler(sh["rpy"], "XYZ").to_matrix().to_4x4())
            wl, wr, _ = (frame @ local).decompose()
            e = wr.to_euler("XYZ")
            shapes.append(dict(sh, center=(wl.x, wl.y, wl.z), rpy=(e.x, e.y, e.z)))
        finally:
            if me is not None:
                bpy.data.meshes.remove(me)
    return shapes


# ----------------------------------------------------------------------------
# Mesh file export (via a temporary object so the user's scene is untouched)
# ----------------------------------------------------------------------------
def export_mesh_file(me, filepath, fmt):
    view_layer = bpy.context.view_layer
    prev_selected = [o for o in view_layer.objects if o.select_get()]
    prev_active = view_layer.objects.active

    tmp_coll = bpy.data.collections.new("__gz_export_tmp__")
    bpy.context.scene.collection.children.link(tmp_coll)
    tmp_obj = bpy.data.objects.new("__gz_export_tmp__", me)
    tmp_coll.objects.link(tmp_obj)
    try:
        view_layer.update()
        for o in view_layer.objects:
            try:
                o.select_set(False)
            except RuntimeError:
                pass
        tmp_obj.select_set(True)
        view_layer.objects.active = tmp_obj

        if fmt == "obj":
            bpy.ops.wm.obj_export(
                filepath=str(filepath),
                export_selected_objects=True,
                apply_modifiers=False,
                export_uv=True,
                export_normals=True,
                export_materials=True,
                export_triangulated_mesh=True,
                path_mode="COPY",            # copy textures next to the mesh
                forward_axis="Y",            # Blender's native axes -> no rotation
                up_axis="Z",
            )
        else:
            bpy.ops.export_scene.gltf(
                filepath=str(filepath),
                export_format="GLB",
                use_selection=True,
                export_apply=False,
            )
    finally:
        bpy.data.objects.remove(tmp_obj)
        bpy.data.collections.remove(tmp_coll)
        for o in view_layer.objects:
            try:
                o.select_set(o in prev_selected)
            except RuntimeError:
                pass
        view_layer.objects.active = prev_active


# ----------------------------------------------------------------------------
# SDF text generation
# ----------------------------------------------------------------------------
def _shape_geometry(sh):
    if sh["type"] == "box":
        x, y, z = sh["size"]
        return f"<box><size>{x:.6f} {y:.6f} {z:.6f}</size></box>"
    if sh["type"] == "sphere":
        return f"<sphere><radius>{sh['radius']:.6f}</radius></sphere>"
    return (f"<cylinder><radius>{sh['radius']:.6f}</radius>"
            f"<length>{sh['length']:.6f}</length></cylinder>")


def model_sdf(name, mesh_file, collision_mode, shapes, sdf_version):
    uri = f"model://{name}/meshes/{mesh_file}"
    if collision_mode == "mesh":
        collision = f"""
      <collision name="collision">
        <geometry>
          <mesh><uri>{uri}</uri></mesh>
        </geometry>
      </collision>"""
    elif collision_mode in ("box", "auto"):
        parts = []
        for i, sh in enumerate(shapes or []):
            c, r = sh["center"], sh["rpy"]
            parts.append(f"""
      <collision name="collision_{i}">
        <pose>{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}</pose>
        <geometry>
          {_shape_geometry(sh)}
        </geometry>
      </collision>""")
        collision = "".join(parts)
    else:
        collision = ""
    return f"""<?xml version="1.0"?>
<sdf version="{sdf_version}">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <mesh><uri>{uri}</uri></mesh>
        </geometry>
      </visual>{collision}
    </link>
  </model>
</sdf>
"""


def model_config(name, sdf_version):
    return f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="{sdf_version}">model.sdf</sdf>
  <author><name>Blender Gazebo exporter</name></author>
  <description>Auto-generated from Blender.</description>
</model>
"""


def system_plugins(s):
    if s.gz_flavor == "gz":
        fn, ns, eng = "gz-sim-{}-system", "gz::sim::systems::{}", "ogre2"
    else:
        fn, ns, eng = "ignition-gazebo-{}-system", "ignition::gazebo::systems::{}", "ogre2"
    items = [("physics", "Physics"), ("user-commands", "UserCommands"),
             ("scene-broadcaster", "SceneBroadcaster"), ("sensors", "Sensors"), ("imu", "Imu")]
    out = []
    for file_key, cls in items:
        extra = f"\n      <render_engine>{eng}</render_engine>\n    " if cls == "Sensors" else ""
        out.append(f'    <plugin filename="{fn.format(file_key)}" name="{ns.format(cls)}">{extra}</plugin>')
    return "\n".join(out)


GROUND_PLANE = """    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

SUN = """    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
"""


def include_block(instances):
    lines = [MARKER_BEGIN, "    <!-- Generated from Blender. Edits between the markers are overwritten. -->"]
    for model, inst, pose in instances:
        lines += [
            "    <include>",
            f"      <uri>model://{model}</uri>",
            f"      <name>{inst}</name>",
            f"      <pose>{pose}</pose>",
            "    </include>",
        ]
    lines.append("    " + MARKER_END)
    return "\n".join(lines)


def fresh_world(s, block):
    return f"""<?xml version="1.0" ?>
<sdf version="{s.sdf_version}">
  <world name="{s.world_name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
{system_plugins(s)}
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.8 0.9 1</background>
      <shadows>true</shadows>
    </scene>
{SUN}{GROUND_PLANE if s.include_ground_plane else ""}
    {block}
  </world>
</sdf>
"""


def write_world(path, s, instances):
    block = include_block(instances)
    if path.exists():
        text = path.read_text()
        if s.backup_world:
            shutil.copy2(path, str(path) + ".bak")
        if MARKER_BEGIN in text and MARKER_END in text:
            b = text.index(MARKER_BEGIN)
            e = text.index(MARKER_END) + len(MARKER_END)
            path.write_text(text[:b] + block + text[e:])
            return "updated (kept your edits outside the markers)"
        path.write_text(fresh_world(s, block))
        return "replaced (existing file had no markers; backup saved as .bak)"
    path.write_text(fresh_world(s, block))
    return "created"


def write_run_script(out_dir, s):
    if s.gz_flavor == "gz":
        env, cmd = "GZ_SIM_RESOURCE_PATH", "gz sim"
    else:
        env, cmd = "IGN_GAZEBO_RESOURCE_PATH", "ign gazebo"
    script = f"""#!/usr/bin/env bash
# Generated by the Blender Gazebo exporter
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export {env}="$DIR/models${{{env}:+:${{{env}}}}}"
exec {cmd} -r "$DIR/worlds/{s.world_name}.sdf" "$@"
"""
    p = out_dir / "run_world.sh"
    p.write_text(script)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ----------------------------------------------------------------------------
# Main export
# ----------------------------------------------------------------------------
@dataclass
class ModelPlan:
    name: str
    kind: str                      # "single" or "group"
    objs: list
    collision: str
    mesh_matrix: Matrix = None     # single: scale-only matrix
    coll: object = None            # group: the Blender collection


def export_world(s):
    s.finalize()
    root = bpy.data.collections.get(s.collection)
    if root is None:
        names = ", ".join(c.name for c in bpy.data.collections) or "(none)"
        raise ValueError(f"Collection '{s.collection}' not found. Available: {names}")

    ensure_object_mode()
    out_dir = Path(s.output_dir).expanduser().resolve()
    models_dir, worlds_dir = out_dir / "models", out_dir / "worlds"
    models_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir.mkdir(parents=True, exist_ok=True)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    merged = set(s.merged_collections)

    # ---- 1. Plan -----------------------------------------------------------
    singles, groups = [], {}
    gather(root, s, merged, singles, groups)
    seen, unique_singles = set(), []
    for o in sorted(singles, key=lambda o: o.name):
        if o.name not in seen:
            seen.add(o.name)
            unique_singles.append(o)

    used_models = set()
    used_instances = {"ground_plane", "sun"}
    plans, instances = [], []

    # bucket objects that are the same thing (same geometry + scale + materials)
    sig_cache, buckets = {}, {}
    for obj in unique_singles:
        loc, rot, scale = obj.matrix_world.decompose()
        key = single_key(obj, scale, depsgraph, sig_cache, s)
        buckets.setdefault(key, []).append((obj, loc, rot.to_euler("XYZ"), scale))

    for members in buckets.values():
        objs = [m[0] for m in members]
        first, scale = objs[0], members[0][3]
        plan = ModelPlan(
            name=unique_name(model_base_name(objs, s), used_models), kind="single", objs=[first],
            collision=first.get("gz_collision", s.collision),
            mesh_matrix=Matrix.Diagonal(Vector((scale.x, scale.y, scale.z, 1.0))),
        )
        plans.append(plan)
        for obj, loc, euler, _ in members:
            inst = unique_name(sanitize(obj.name), used_instances)
            instances.append((plan.name, inst, fmt_pose(loc, euler)))

    for cname in sorted(groups):
        coll, objs = groups[cname]
        plan = ModelPlan(
            name=unique_name(sanitize(cname), used_models), kind="group", objs=objs,
            collision=coll.get("gz_collision", s.collision), coll=coll,
        )
        plans.append(plan)
        inst = unique_name(sanitize(cname), used_instances)
        instances.append((plan.name, inst, "0 0 0 0 0 0"))

    for p in plans:
        if p.collision not in ("mesh", "box", "auto", "none"):
            raise ValueError(f"Invalid gz_collision '{p.collision}' on '{p.name}'")

    # refuse to overwrite hand-made model folders
    for p in plans:
        d = models_dir / p.name
        if d.exists() and not (d / MODEL_MARKER).exists():
            raise RuntimeError(
                f"'{d}' exists but was not created by this exporter. "
                f"Rename the Blender object/mesh or move that folder."
            )

    # ---- 2. Write models ---------------------------------------------------
    ext = s.mesh_format
    written, skipped = set(), set()
    for p in plans:
        model_dir = models_dir / p.name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        meshes_dir = model_dir / "meshes"
        meshes_dir.mkdir(parents=True)

        if p.kind == "single":
            me = evaluated_mesh(p.objs[0], depsgraph)
            apply_object_materials(me, p.objs[0])
            transform_mesh(me, p.mesh_matrix)
        else:
            me = build_merged_mesh(p.objs, depsgraph, p.name)

        if me is None or len(me.polygons) == 0:
            print(f"[skip] '{p.name}' has no faces")
            if me is not None:
                bpy.data.meshes.remove(me)
            shutil.rmtree(model_dir)
            skipped.add(p.name)
            continue

        try:
            shapes = None
            if p.collision in ("box", "auto"):
                auto = p.collision == "auto"
                if p.kind == "single":
                    shapes = [fit_shape(mesh_coords(me), auto)]
                else:
                    shapes = object_shapes(p.objs, depsgraph, auto)
            mesh_file = f"{p.name}.{ext}"
            export_mesh_file(me, meshes_dir / mesh_file, ext)
        finally:
            bpy.data.meshes.remove(me)

        (model_dir / "model.sdf").write_text(model_sdf(p.name, mesh_file, p.collision, shapes, s.sdf_version))
        (model_dir / "model.config").write_text(model_config(p.name, s.sdf_version))
        (model_dir / MODEL_MARKER).write_text(json.dumps({"world": s.world_name}))
        written.add(p.name)
        used = ",".join(sorted({sh["type"] for sh in shapes})) if shapes else p.collision
        print(f"[model] {p.name}  ({p.kind}, collision={p.collision}: {used})")

    instances = [i for i in instances if i[0] not in skipped]

    # ---- 3. Remove stale models from earlier exports of this world --------
    removed = []
    if s.remove_stale:
        for d in models_dir.iterdir():
            marker = d / MODEL_MARKER
            if d.is_dir() and marker.exists() and d.name not in written:
                try:
                    owner = json.loads(marker.read_text()).get("world")
                except ValueError:
                    owner = None
                if owner == s.world_name:
                    shutil.rmtree(d)
                    removed.append(d.name)

    # ---- 4. World + helper script -----------------------------------------
    world_path = worlds_dir / f"{s.world_name}.sdf"
    world_status = write_world(world_path, s, instances)
    write_run_script(out_dir, s)

    env = "GZ_SIM_RESOURCE_PATH" if s.gz_flavor == "gz" else "IGN_GAZEBO_RESOURCE_PATH"
    summary = (
        f"\nDONE. World {world_status}: {world_path}\n"
        f"  models written: {len(written)}   instances in world: {len(instances)}   "
        f"stale removed: {len(removed)}\n"
        f"  Run:  {out_dir / 'run_world.sh'}\n"
        f"  Manual: export {env}={models_dir}:${env}  &&  "
        f"{'gz sim' if s.gz_flavor == 'gz' else 'ign gazebo'} -r {world_path}\n"
        f"  ROS 2: add 'install(DIRECTORY models worlds DESTINATION share/${{PROJECT_NAME}})' to CMakeLists.txt\n"
        f"         and set {env} to <share>/models in your launch file."
    )
    print(summary)
    return summary


# ----------------------------------------------------------------------------
# Blender operator (folder-picker dialog) and CLI
# ----------------------------------------------------------------------------
class EXPORT_OT_gazebo_world(bpy.types.Operator):
    """Export a collection as a Gazebo world (SDF + models + meshes)"""
    bl_idname = "export_scene.gazebo_world"
    bl_label = "Export Gazebo World"

    directory: bpy.props.StringProperty(
        name="Output folder", subtype="DIR_PATH",
        default=os.path.expanduser(SETTINGS.output_dir))
    world_name: bpy.props.StringProperty(name="World name", default=SETTINGS.world_name)
    collection: bpy.props.StringProperty(name="Collection", default=SETTINGS.collection)
    merged_collections: bpy.props.StringProperty(
        name="Merge collections", default=",".join(SETTINGS.merged_collections),
        description="Comma-separated collection names baked into one mesh each")
    mesh_format: bpy.props.EnumProperty(
        name="Mesh format", default=SETTINGS.mesh_format,
        items=[("obj", "OBJ + MTL", "Most compatible"), ("glb", "glTF binary (.glb)", "PBR materials")])
    collision: bpy.props.EnumProperty(
        name="Collision", default=SETTINGS.collision,
        items=[("mesh", "Mesh (exact)", ""), ("box", "Box (fast)", ""),
               ("auto", "Auto best-fit box/cylinder/sphere", ""), ("none", "None", "")])
    gz_flavor: bpy.props.EnumProperty(
        name="Gazebo", default=SETTINGS.gz_flavor,
        items=[("gz", "Garden / Harmonic / Ionic", ""), ("ignition", "Fortress", "")])
    include_ground_plane: bpy.props.BoolProperty(name="Ground plane", default=SETTINGS.include_ground_plane)
    include_hidden: bpy.props.BoolProperty(name="Include hidden objects", default=SETTINGS.include_hidden)
    remove_stale: bpy.props.BoolProperty(name="Remove stale models", default=SETTINGS.remove_stale)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        s = replace(
            SETTINGS,
            output_dir=self.directory,
            world_name=self.world_name,
            collection=self.collection,
            merged_collections=[c.strip() for c in self.merged_collections.split(",") if c.strip()],
            mesh_format=self.mesh_format,
            collision=self.collision,
            gz_flavor=self.gz_flavor,
            include_ground_plane=self.include_ground_plane,
            include_hidden=self.include_hidden,
            remove_stale=self.remove_stale,
        )
        try:
            export_world(s)
        except Exception as exc:  # show the error in Blender's UI
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Gazebo world exported to {Path(s.output_dir).expanduser()}")
        return {"FINISHED"}


def _menu_func(self, context):
    self.layout.operator(EXPORT_OT_gazebo_world.bl_idname, text="Gazebo World (.sdf)")


def register():
    try:
        bpy.types.TOPBAR_MT_file_export.remove(_menu_func)
        bpy.utils.unregister_class(EXPORT_OT_gazebo_world)
    except (RuntimeError, ValueError):
        pass
    bpy.utils.register_class(EXPORT_OT_gazebo_world)
    bpy.types.TOPBAR_MT_file_export.append(_menu_func)


def settings_from_cli():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="export_gazebo_world")
    ap.add_argument("--output-dir")
    ap.add_argument("--world-name")
    ap.add_argument("--collection")
    ap.add_argument("--merge", help="comma-separated collections to merge into one mesh each")
    ap.add_argument("--format", choices=["obj", "glb"])
    ap.add_argument("--collision", choices=["mesh", "box", "auto", "none"])
    ap.add_argument("--flavor", choices=["gz", "ignition"])
    ap.add_argument("--include-hidden", action="store_true")
    ap.add_argument("--no-ground-plane", action="store_true")
    ap.add_argument("--keep-stale", action="store_true")
    a = ap.parse_args(argv)

    s = replace(SETTINGS, merged_collections=list(SETTINGS.merged_collections))
    if a.output_dir: s.output_dir = a.output_dir
    if a.world_name: s.world_name = a.world_name
    if a.collection: s.collection = a.collection
    if a.merge is not None: s.merged_collections = [c.strip() for c in a.merge.split(",") if c.strip()]
    if a.format: s.mesh_format = a.format
    if a.collision: s.collision = a.collision
    if a.flavor: s.gz_flavor = a.flavor
    if a.include_hidden: s.include_hidden = True
    if a.no_ground_plane: s.include_ground_plane = False
    if a.keep_stale: s.remove_stale = False
    return s


if __name__ == "__main__":
    if bpy.app.background:
        export_world(settings_from_cli())
    else:
        register()
        bpy.ops.export_scene.gazebo_world("INVOKE_DEFAULT")
