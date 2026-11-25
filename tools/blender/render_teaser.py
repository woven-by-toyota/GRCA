"""Path-traced (Blender Cycles) render of a GRCA teaser preset, from the viewer's exact camera.

    blender -b --factory-startup -P tools/blender/render_teaser.py -- \
        --preset presets/benchmark_config_vis_teaser_SanMiguel.json --out render.png [options]

Reads the preset's scene file (OBJ paths, scale, offset) and its first vis_shot (w_cam_position, w_cam_forward,
w_cam_fov) plus vis_window_size, so the render lines up with the viewer's captures. The viewer is right-handed and
y-up; Blender's OBJ importer maps OBJ (x, y, z) to Blender (x, -z, y), and the camera is mapped the same way.
Textures are read from the paths in each MTL (the textured mesh.zip layout); --texdirs adds folders searched by
file name for any that are missing. tools/blender/render_teasers.sh holds the settings used for the paper.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--grca", default=ROOT, help="GRCA root (scene and mesh paths resolve against it)")
ap.add_argument("--preset", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mesh-root", help="replace the leading 'mesh/' of scene OBJ paths with this folder")
ap.add_argument("--texdirs", nargs="*", default=[], help="folders searched by file name for textures the MTL misses")
ap.add_argument("--samples", type=int, default=512)
ap.add_argument("--res", type=int, default=100, help="resolution percentage")
ap.add_argument("--sun", nargs="+", default=["auto"],
                help="'auto' (most direct light on what the camera sees) or ELEVATION AZIMUTH in degrees; "
                     "azimuth is relative to the camera heading (180 = behind the camera)")
ap.add_argument("--sun-strength", type=float, default=10.0)
ap.add_argument("--sun-color", nargs=3, type=float, default=[1.0, 0.93, 0.84])
ap.add_argument("--sky-strength", type=float, default=0.4)
ap.add_argument("--exposure", type=float, default=0.0)
ap.add_argument("--look", default="AgX - Medium High Contrast")
ap.add_argument("--min-roughness", type=float, default=0.0, help="raise untextured roughness to at least this")
ap.add_argument("--pixel-art", action="store_true", help="nearest-neighbour texture filtering (Minecraft scenes)")
ap.add_argument("--bevel", type=float, default=0.0, help="rounded-edge shading radius in metres (Cycles Bevel)")
ap.add_argument("--tex-bump", type=float, default=0.0, help="bump strength from each base-colour texture")
ap.add_argument("--haze", type=float, default=0.0,
                help="scattering density (1/m) of a volume box around the scene (a world volume would absorb sun and sky)")
ap.add_argument("--no-lidar", action="store_true", help="leave out the LiDAR cylinder")
ap.add_argument("--texture-limit", default="OFF", help="Cycles render texture limit, e.g. 2048")
ap.add_argument("--save-blend")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def rel(p):
    return p if os.path.isabs(p) else os.path.join(a.grca, p)


preset = json.load(open(rel(a.preset)))
scene_cfg = json.load(open(rel(preset["benchmark_scenes"][0])))
shot = preset["vis_shots"][0]
eye, fwd = shot["w_cam_position"], shot["w_cam_forward"]
fov = shot.get("w_cam_fov", preset.get("w_cam_fov", 45.0))
W, H = preset.get("vis_window_size", [1000, 1000])


def to_blender(v):
    return Vector((v[0], -v[2], v[1]))


# --- geometry ------------------------------------------------------------------------------------------------
for m in scene_cfg["static_meshes"]:
    path = m["file"]
    if a.mesh_root and path.startswith("mesh/"):
        path = os.path.join(a.mesh_root, path[len("mesh/"):])
    path = rel(path)
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, global_scale=m.get("scale", 1.0), forward_axis='NEGATIVE_Z', up_axis='Y')
    off = to_blender(m.get("w_position", [0, 0, 0]))
    if off.length > 0:
        for o in set(bpy.data.objects) - before:
            o.location += off
    print(f"[render] imported {path}  scale {m.get('scale', 1.0)}", flush=True)

# --- textures ------------------------------------------------------------------------------------------------
index = {}
for d in a.texdirs:
    for r, _, fs in os.walk(d):
        for f in fs:
            index.setdefault(f.lower(), os.path.join(r, f))
fixed = missing = 0
for img in bpy.data.images:
    if img.source != 'FILE' or img.size[0] > 0:   # size > 0: loaded (MTL paths may use backslashes)
        continue
    base = os.path.basename(img.filepath.replace("\\", "/")).lower()
    stem = os.path.splitext(base)[0]
    cand = next((index[k] for k in (base, stem + ".png", stem + ".dds", stem + ".jpg") if k in index), None)
    if cand:
        img.filepath = cand
        img.reload()
        fixed += 1
    else:
        missing += 1
print(f"[render] images {len(bpy.data.images)}, remapped {fixed}, still missing {missing}", flush=True)

# --- materials: filtering, roughness, texture bump, bevelled edges -------------------------------------------
for mat in bpy.data.materials:
    if not mat.use_nodes:
        continue
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type == 'TEX_IMAGE' and a.pixel_art:
            n.interpolation = 'Closest'
        if n.type != 'BSDF_PRINCIPLED':
            continue
        if a.min_roughness > 0 and not n.inputs['Roughness'].is_linked:
            n.inputs['Roughness'].default_value = max(n.inputs['Roughness'].default_value, a.min_roughness)
        nin = n.inputs['Normal']
        prev = nin.links[0].from_socket if nin.is_linked else None
        bc = n.inputs['Base Color']
        if a.tex_bump > 0 and bc.is_linked and bc.links[0].from_node.type == 'TEX_IMAGE':
            bump = nt.nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = a.tex_bump
            bump.inputs['Distance'].default_value = 0.02
            nt.links.new(bc.links[0].from_socket, bump.inputs['Height'])
            if prev is not None:
                nt.links.new(prev, bump.inputs['Normal'])
            prev = bump.outputs['Normal']
        if a.bevel > 0:
            bev = nt.nodes.new('ShaderNodeBevel')
            bev.samples = 8
            bev.inputs['Radius'].default_value = a.bevel
            if prev is not None:
                nt.links.new(prev, bev.inputs['Normal'])
            prev = bev.outputs['Normal']
        if prev is not None and not (nin.is_linked and nin.links[0].from_socket == prev):
            nt.links.new(prev, nin)

# --- camera --------------------------------------------------------------------------------------------------
cam = bpy.data.cameras.new("ViewerCam")
cam.sensor_fit = 'VERTICAL'
cam.angle_y = math.radians(fov)
cam.clip_start = 0.05
cam.clip_end = 1.0e5
cam_obj = bpy.data.objects.new("ViewerCam", cam)
sc.collection.objects.link(cam_obj)
sc.camera = cam_obj
f_b = to_blender(fwd).normalized()
cam_obj.location = to_blender(eye)
cam_obj.rotation_mode = 'QUATERNION'
cam_obj.rotation_quaternion = f_b.to_track_quat('-Z', 'Y')
bpy.context.view_layer.update()

# --- LiDAR: the viewer's cylinder (radius = height = 0.15 m, base at the LiDAR origin, axis = the LiDAR's up) -------
if not a.no_lidar and shot.get("robot_w_positions"):
    robots = json.load(open(rel(preset["benchmark_robots"][0])))
    for rp, robot in zip(shot["robot_w_positions"], robots):
        f_r, u_r = Vector(robot.get("w_forward", [0, 0, 1])).normalized(), Vector(robot.get("w_up", [0, 1, 0])).normalized()
        r_r = u_r.cross(f_r).normalized()           # robot local x (z = forward, y = up)
        for lid in robot["lidars"]:
            lp, lu = Vector(lid.get("position", [0, 0, 0])), Vector(lid.get("up", [0, 1, 0]))
            pos = Vector(rp) + r_r * lp.x + u_r * lp.y + f_r * lp.z
            axis = (r_r * lu.x + u_r * lu.y + f_r * lu.z).normalized()
            base, axis_b = to_blender(pos), to_blender(axis).normalized()
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.15, depth=0.15, location=base + axis_b * 0.075)
            cyl = bpy.context.active_object
            cyl.name = "LiDAR"
            cyl.rotation_mode = 'QUATERNION'
            cyl.rotation_quaternion = axis_b.to_track_quat('Z', 'Y')
            bpy.ops.object.shade_smooth()
            lm = bpy.data.materials.new("LiDAR")
            lm.use_nodes = True
            bsdf = lm.node_tree.nodes["Principled BSDF"]
            bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.35
            cyl.data.materials.append(lm)
            print(f"[render] LiDAR cylinder at {[round(x, 3) for x in pos]} (viewer coords), axis {[round(x, 3) for x in axis]}", flush=True)

# --- sun direction -------------------------------------------------------------------------------------------
heading = Vector((f_b.x, f_b.y, 0.0))
heading = heading.normalized() if heading.length > 1e-6 else Vector((0.0, 1.0, 0.0))


def sun_dir(el_deg, az_deg):
    el, az = math.radians(el_deg), math.radians(az_deg)
    hx = heading.x * math.cos(az) - heading.y * math.sin(az)
    hy = heading.x * math.sin(az) + heading.y * math.cos(az)
    return Vector((hx * math.cos(el), hy * math.cos(el), math.sin(el))).normalized()


def auto_sun():
    """Sun direction that lights the most of what the camera sees: cast a 48 x 36 grid of camera rays, then for
    each candidate direction sum max(0, n . s) over the hit points that see the sun unoccluded."""
    dg = bpy.context.evaluated_depsgraph_get()
    m3, o = cam_obj.matrix_world.to_3x3(), cam_obj.matrix_world.translation
    ty = math.tan(cam.angle_y / 2.0)
    tx = ty * W / H
    pts = []
    for j in range(36):
        for i in range(48):
            d = (m3 @ Vector((((i + 0.5) / 48 * 2 - 1) * tx, ((j + 0.5) / 36 * 2 - 1) * ty, -1.0))).normalized()
            hit, loc, nrm, _, _, _ = sc.ray_cast(dg, o, d)
            if hit:
                pts.append((loc, -nrm if nrm.dot(d) > 0 else nrm, 1e-4 * max(1.0, (loc - o).length)))
    scores = []
    for el in (30, 40, 50, 60, 70, 80):
        for az in range(0, 360, 20):
            s = sun_dir(el, az)
            lit = 0.0
            for loc, n, eps in pts:
                c = n.dot(s)
                if c > 0 and not sc.ray_cast(dg, loc + n * eps, s)[0]:
                    lit += c
            scores.append((lit / max(1, len(pts)), el, az))
    scores.sort(reverse=True)
    print("[render] auto sun, best (direct light, elevation, azimuth):",
          [(round(x, 3), e, z) for x, e, z in scores[:5]], flush=True)
    # Straight-behind light is flat: among directions within 3% of the best, take the one most to the side.
    near = [t for t in scores if t[0] >= 0.97 * scores[0][0]]
    _, el, az = max(near, key=lambda t: (abs(t[2] - 180), t[0]))
    return el, az


if a.sun[0] == "auto":
    sun_el, sun_az = auto_sun()
else:
    sun_el, sun_az = float(a.sun[0]), float(a.sun[1])
to_sun = sun_dir(sun_el, sun_az)
print(f"[render] sun elevation {sun_el} azimuth {sun_az} (relative to camera heading)", flush=True)

# --- light: physical sky (no disc) plus a sun lamp; optional haze --------------------------------------------
world = bpy.data.worlds.new("World")
sc.world = world
world.use_nodes = True
wn, wl = world.node_tree.nodes, world.node_tree.links
wn.clear()
sky = wn.new("ShaderNodeTexSky")
for sky_type in ('MULTIPLE_SCATTERING', 'NISHITA', 'HOSEK_WILKIE'):   # newest physical sky first
    try:
        sky.sky_type = sky_type
        break
    except TypeError:
        continue
sky.sun_disc = False
sky.sun_elevation = math.radians(sun_el)
sky.sun_rotation = math.atan2(to_sun.x, to_sun.y)
bg = wn.new("ShaderNodeBackground")
bg.inputs['Strength'].default_value = a.sky_strength
out = wn.new("ShaderNodeOutputWorld")
wl.new(sky.outputs['Color'], bg.inputs['Color'])
wl.new(bg.outputs['Background'], out.inputs['Surface'])
if a.haze > 0:
    corners = [o.matrix_world @ Vector(c) for o in bpy.data.objects if o.type == 'MESH' for c in o.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    pad = 0.05 * (hi - lo).length
    bpy.ops.mesh.primitive_cube_add(location=(lo + hi) / 2)
    box = bpy.context.active_object
    box.name = "Haze"
    box.scale = (hi - lo) / 2 + Vector((pad, pad, pad))
    hm = bpy.data.materials.new("Haze")
    hm.use_nodes = True
    hn = hm.node_tree.nodes
    hn.remove(hn["Principled BSDF"])
    vol = hn.new("ShaderNodeVolumeScatter")
    vol.inputs['Density'].default_value = a.haze
    vol.inputs['Anisotropy'].default_value = 0.6
    hm.node_tree.links.new(vol.outputs['Volume'], hn["Material Output"].inputs['Volume'])
    box.data.materials.append(hm)
    print(f"[render] haze box {[round(x, 1) for x in lo]} .. {[round(x, 1) for x in hi]}, density {a.haze}", flush=True)

sun = bpy.data.lights.new("Sun", 'SUN')
sun.energy = a.sun_strength
sun.color = a.sun_color
sun.angle = math.radians(0.6)
sun_obj = bpy.data.objects.new("Sun", sun)
sc.collection.objects.link(sun_obj)
sun_obj.rotation_mode = 'QUATERNION'
sun_obj.rotation_quaternion = (-to_sun).to_track_quat('-Z', 'Y')

# --- render --------------------------------------------------------------------------------------------------
sc.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
for dev_type in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
    try:
        prefs.compute_device_type = dev_type
    except TypeError:
        continue
    prefs.get_devices()
    if any(d.type == dev_type for d in prefs.devices):
        break
for d in prefs.devices:
    d.use = (d.type == prefs.compute_device_type)
sc.cycles.device = 'GPU' if any(d.use for d in prefs.devices) else 'CPU'
sc.cycles.samples = a.samples
sc.cycles.use_denoising = True
sc.cycles.max_bounces = 8
if a.texture_limit != "OFF":
    sc.render.use_simplify = True
    sc.cycles.texture_limit_render = a.texture_limit
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.resolution_percentage = a.res
sc.view_settings.view_transform = 'AgX'
sc.view_settings.exposure = a.exposure
try:
    sc.view_settings.look = a.look
except TypeError as e:
    print(f"[render] look {a.look!r} not available: {e}", flush=True)
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.filepath = rel(a.out)
if a.save_blend:
    bpy.ops.wm.save_as_mainfile(filepath=rel(a.save_blend))
print(f"[render] {prefs.compute_device_type if sc.cycles.device == 'GPU' else 'CPU'}, camera {list(cam_obj.location)}, "
      f"fov {fov} -> {sc.render.filepath} ({W}x{H} at {a.res}%, {a.samples} spp)", flush=True)
bpy.ops.render.render(write_still=True)
print("[render] done", flush=True)
